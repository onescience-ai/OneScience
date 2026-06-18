# train_graphvit_eagle.py
import os
import sys
import logging
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from torch.utils.data import DataLoader
from onescience.distributed.manager import DistributedManager
from onescience.utils.YParams import YParams
from onescience.datapipes.cfd import EagleDatapipe
from onescience.models.graphvit import GraphViT


def save_best_model(model, optimizer, scheduler, ckp_dir, model_name="best_model.pth"):
    """保存当前最优模型"""
    if not os.path.exists(ckp_dir):
        os.makedirs(ckp_dir, exist_ok=True)

    model_to_save = model.module if hasattr(model, "module") else model
    state = {
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scheduler:
        state["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(state, os.path.join(ckp_dir, model_name))


def load_best_model(model, ckp_dir, device, model_name="best_model.pth"):
    """加载最优模型权重"""
    ckpt_path = os.path.join(ckp_dir, model_name)
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        model_to_load = model.module if hasattr(model, "module") else model
        try:
            model_to_load.load_state_dict(checkpoint["model_state_dict"])
        except KeyError:
            model_to_load.load_state_dict(checkpoint)
        logging.info(f"Successfully loaded model from {ckpt_path}")
    else:
        logging.warning(f"Checkpoint file not found: {ckpt_path}. Model training from scratch.")


def setup_logging(rank):
    level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    logging.getLogger("torch.distributed").setLevel(logging.WARNING)
    logger = logging.getLogger()
    logger.setLevel(level)
    return logger


def get_loss(velocity, pressure, output, state_hat, target, mask, alpha):
    """计算速度、压力与联合损失"""
    velocity = velocity[:, 1:]
    pressure = pressure[:, 1:]

    velocity_hat = state_hat[:, 1:, :, :2]
    pressure_hat = state_hat[:, 1:, :, 2:]

    mask = mask[:, 1:].unsqueeze(-1)

    rmse_velocity = torch.sqrt(
        ((velocity * mask - velocity_hat * mask) ** 2).mean(dim=(-1))
    )
    loss_velocity = torch.mean(rmse_velocity)

    rmse_pressure = torch.sqrt(
        ((pressure * mask - pressure_hat * mask) ** 2).mean(dim=(-1))
    )
    loss_pressure = torch.mean(rmse_pressure)

    mse = nn.MSELoss()
    loss = mse(target[..., :2] * mask, output[..., :2] * mask) + alpha * mse(
        target[..., 2:] * mask, output[..., 2:] * mask
    )

    return {
        "loss": loss,
        "MSE_velocity": loss_velocity,
        "MSE_pressure": loss_pressure,
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    epoch: int,
    device: torch.device,
    alpha: float,
    manager: DistributedManager,
):
    """验证阶段"""
    model.eval()
    total_loss, cpt = 0.0, 0

    with torch.no_grad():
        pbar = tqdm(
            dataloader,
            desc="Validation",
            disable=(manager.rank != 0),
            dynamic_ncols=True,
            mininterval=0,
            miniters=1,
            leave=True,
            file=sys.stdout,
        )
        for x in pbar:
            if not x:
                continue

            mesh_pos = x["mesh_pos"].to(device)
            edges = x["edges"].to(device).long()
            velocity = x["velocity"].to(device)
            pressure = x["pressure"].to(device)
            node_type = x["node_type"].to(device)
            mask = x["mask"].to(device)
            clusters = x["cluster"].to(device).long()
            clusters_mask = x["cluster_mask"].to(device).long()

            state = torch.cat([velocity, pressure], dim=-1)

            state_hat, output, target = model(
                mesh_pos,
                edges,
                state,
                node_type,
                clusters,
                clusters_mask,
                apply_noise=False,
            )

            dataset = dataloader.dataset
            state_hat[..., :2], state_hat[..., 2:] = dataset.denormalize(
                state_hat[..., :2], state_hat[..., 2:]
            )
            velocity, pressure = dataset.denormalize(velocity, pressure)

            costs = get_loss(
                velocity, pressure, output, state_hat, target, mask, alpha
            )

            if manager.world_size > 1:
                dist.all_reduce(costs["loss"], op=dist.ReduceOp.AVG)

            total_loss += costs["loss"].item()
            cpt += 1

        if manager.world_size > 1:
            dist.barrier()

    return total_loss / cpt if cpt > 0 else 0.0


def main():
    DistributedManager.initialize()
    manager = DistributedManager()
    logger = setup_logging(manager.rank)

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    config_file_path = "conf/graphvit_eagle.yaml"
    cfg = YParams(config_file_path, "model")
    cfg_data = YParams(config_file_path, "datapipe")
    cfg_train = YParams(config_file_path, "training")

    model_name = cfg.name
    if manager.rank == 0:
        logger.info(f"=====  Preparing model: {model_name} =====")
        logger.info(f"Loading config from: {config_file_path}")

    logger.info("Initializing datapipe...")
    datapipe = EagleDatapipe(params=cfg_data, distributed=(manager.world_size > 1))
    train_dataloader, train_sampler = datapipe.train_dataloader()
    val_dataloader, val_sampler = datapipe.val_dataloader()

    device = manager.device

    logger.info(f"Initializing model architecture: {model_name}")
    model = GraphViT(state_size=cfg.state_size, w_size=cfg.w_size).to(device)

    if manager.rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"Model: {model_name}, Trainable Params: {total_params / 1e6:.2f}M"
        )

    if manager.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[manager.local_rank],
            output_device=manager.local_rank,
            find_unused_parameters=True,
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg_train.lr)
    scheduler = None
    loss_alpha = cfg_train.loss_alpha

    checkpoint_dir = cfg_train.checkpoint_dir
    if manager.rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)

    best_valid_loss = 1.0e6
    best_epoch = 0

    logger.info("Starting training...")
    for epoch in range(cfg_train.max_epoch):
        epoch_start_time = time.time()
        epoch_num = epoch + 1

        if manager.world_size > 1:
            train_sampler.set_epoch(epoch)
            if val_sampler:
                val_sampler.set_epoch(epoch)

        model.train()
        train_loss, train_cpt = 0.0, 0

        pbar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch_num}/{cfg_train.max_epoch} Training",
            disable=(manager.rank != 0),
            dynamic_ncols=True,
            mininterval=0,
            miniters=1,
            leave=True,
            file=sys.stdout,
        )
        total_batches = len(train_dataloader)

        for i, x in enumerate(pbar):
            batch_num = i + 1
            if not x:
                continue

            mesh_pos = x["mesh_pos"].to(device)
            edges = x["edges"].to(device).long()
            velocity = x["velocity"].to(device)
            pressure = x["pressure"].to(device)
            node_type = x["node_type"].to(device)
            mask = x["mask"].to(device)
            clusters = x["cluster"].to(device).long()
            clusters_mask = x["cluster_mask"].to(device).long()

            state = torch.cat([velocity, pressure], dim=-1)

            state_hat, output, target = model(
                mesh_pos,
                edges,
                state,
                node_type,
                clusters,
                clusters_mask,
                apply_noise=True,
            )

            state_hat[..., :2], state_hat[..., 2:] = train_dataloader.dataset.denormalize(
                state_hat[..., :2], state_hat[..., 2:]
            )
            velocity, pressure = train_dataloader.dataset.denormalize(
                velocity, pressure
            )

            costs = get_loss(
                velocity, pressure, output, state_hat, target, mask, loss_alpha
            )

            optimizer.zero_grad()
            costs["loss"].backward()
            optimizer.step()

            train_loss += costs["loss"].item()
            train_cpt += 1
            if manager.rank == 0:
                loss = costs["loss"].item()
                pbar.set_postfix(loss=f"{loss:.6f}")
                logger.info(
                    f"Epoch [{epoch_num}/{cfg_train.max_epoch}] "
                    f"Batch [{batch_num}/{total_batches}] | Loss: {loss:.6f}"
                )

        train_loss /= train_cpt if train_cpt > 0 else 1.0

        valid_loss = validate(
            model, val_dataloader, epoch, device, loss_alpha, manager
        )

        if manager.rank == 0:
            epoch_time = time.time() - epoch_start_time
            logger.info(
                f"Epoch [{epoch + 1}/{cfg_train.max_epoch}] | Time: {epoch_time:.2f}s | "
                f"Train Loss: {train_loss:.6f} | Valid Loss: {valid_loss:.6f}"
            )

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_epoch = epoch
                save_best_model(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    ckp_dir=checkpoint_dir,
                    model_name="best_model.pth",
                )
                logger.info(
                    "   -> New best validation loss. Checkpoint saved as best_model.pth."
                )

            if epoch - best_epoch > cfg_train.patience:
                logger.warning(
                    f"Validation loss has not improved for {cfg_train.patience} epochs. "
                    "Stopping training."
                )
                break

        if manager.world_size > 1:
            dist.barrier()

    if manager.rank == 0:
        logger.info("=====  Training finished. Starting final validation... =====")

        final_model = GraphViT(
            state_size=cfg.state_size, w_size=cfg.w_size
        ).to(device)

        load_best_model(
            model=final_model,
            ckp_dir=checkpoint_dir,
            device=device,
            model_name="best_model.pth",
        )

        test_dataloader, _ = datapipe.test_dataloader()
        final_test_loss = validate(
            final_model,
            test_dataloader,
            best_epoch,
            device,
            loss_alpha,
            manager,
        )

        logger.info(
            f"=====  Final Test Loss (from best model at epoch {best_epoch}): "
            f"{final_test_loss:.6f} ====="
        )

    manager.cleanup()


if __name__ == "__main__":
    main()
