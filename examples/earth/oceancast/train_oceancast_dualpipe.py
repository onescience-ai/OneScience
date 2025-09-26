"""
This Code is used to train dualpipe_pipeline_parallel mode of oceancast

"""

import logging
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from onescience.distributed.comm import set_p2p_tensor_dtype, set_p2p_tensor_shapes
from onescience.models.afno.afnonet_dualpipe import build_dualpipe_model
from onescience.utils.fcn.darcy_loss_ocean import LpLoss as loss_function
from onescience.utils.fcn.data_loader_ocean import get_data_loader
from onescience.utils.fcn.YParams import YParams

mp.set_start_method("spawn", force=True)


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger()

    current_path = os.getcwd()
    config_file_path = os.path.join(
        current_path, "conf/oceancast.yaml")
    params = YParams(config_file_path, "afno_backbone")

    world_rank = 0
    local_rank = 0
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    params["world_size"] = world_size
    params["pipeline_group_size"] = 4
    params["batch_size"] = 8
    world_rank = int(os.environ["RANK"])
    dist.init_process_group(
        "nccl", rank=world_rank, world_size=world_size)

    is_first_rank = (world_rank %
                     params.pipeline_group_size) == 0
    pipeline_group_rank = world_rank % params.pipeline_group_size
    pipeline_group_size = params.pipeline_group_size

    # ✅ 使用 Dualpipe 模型替换原始 AFNONet 构建
    torch.set_default_device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)

    batch_size = params["batch_size"]
    num_chunks = 8
    params["chunks"] = num_chunks
    micro_batch_size = batch_size // num_chunks

    set_p2p_tensor_shapes([(micro_batch_size, 40, 90, 768)])
    set_p2p_tensor_dtype(torch.float32)

    model = build_dualpipe_model(
        params, pipeline_group_rank, pipeline_group_size)
    # train_data_loader, train_dataset, train_sampler = get_data_loader(params, dist.is_initialized(), mode='train')
    # valid_data_loader, valid_dataset, valid_sampler = get_data_loader(params, dist.is_initialized(), mode='valid')
    train_data_loader, train_dataset, train_sampler = get_data_loader(
        params, False, mode="train"
    )
    valid_data_loader, valid_dataset, valid_sampler = get_data_loader(
        params, False, mode="valid"
    )

    mask = torch.load(params.maskpath)
    def loss_obj(x, y): return loss_function(
        mask.to(y.device))(x, y)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=params.lr)
    scheduler = get_scheduler(optimizer, params)

    best_valid_loss = float("inf")
    best_loss_epoch = 0
    model_path = params.model_path
    os.makedirs(model_path, exist_ok=True)
    model_name = f"{model_path}/model_dp.pth"
    train_loss_file = f"{model_path}/trloss.npy"
    valid_loss_file = f"{model_path}/valoss.npy"
    if world_rank == 0:
        print(f"weights and loss are saved at {model_path}")

    train_losses = np.empty((0,), dtype=np.float32)
    valid_losses = np.empty((0,), dtype=np.float32)

    if world_rank == 0:
        logger.info(f"start training {params['name']}...")
    params["max_epochs"] = 1
    for epoch in range(params.max_epochs):
        t1 = time.perf_counter()
        model.train()
        train_loss = 0
        for iteration_idx, data in enumerate(train_data_loader):
            input_tensor, label_tensor = data
            input_tensor = input_tensor.to(
                f"cuda:{local_rank}", dtype=torch.float)
            label_tensor = label_tensor.to(
                f"cuda:{local_rank}", dtype=torch.float)
            if not is_first_rank:
                input_tensor = None
                label_tensor = None

            optimizer.zero_grad()
            loss, _ = model.step(
                input_tensor,
                num_chunks=params.chunks,
                criterion=loss_obj,
                labels=label_tensor,
                return_outputs=False,
            )
            if loss is not None:
                optimizer.step()
                train_loss += loss.mean().item()
            if world_rank == 0 and iteration_idx % (len(train_data_loader) // 8) == 0:
                logger.info(
                    f"Epoch [{epoch + 1}/{params.max_epochs}], Train MiniBatch {iteration_idx}/{len(train_data_loader)} done..."
                )

        train_loss /= len(train_data_loader)

        model.eval()
        valid_loss = 0

        with torch.no_grad():
            for iteration_idx, data in enumerate(valid_data_loader):
                input_tensor, label_tensor = data
                input_tensor = input_tensor.to(
                    f"cuda:{local_rank}", dtype=torch.float)
                label_tensor = label_tensor.to(
                    f"cuda:{local_rank}", dtype=torch.float)
                if not is_first_rank:
                    input_tensor = None
                    label_tensor = None

                loss, _ = model.step(
                    input_tensor,
                    num_chunks=params.chunks,
                    criterion=loss_obj,
                    labels=label_tensor,
                    return_outputs=False,
                )
                if loss is not None:
                    valid_loss += loss.mean().item()

                if (
                    world_rank == 0
                    and iteration_idx % (len(valid_data_loader) // 3) == 0
                ):
                    logger.info(
                        f"Epoch [{epoch + 1}/{params.max_epochs}], Valid MiniBatch {iteration_idx}/{len(valid_data_loader)} done..."
                    )

        valid_loss /= len(valid_data_loader)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_loss_epoch = epoch
            if world_rank == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    best_valid_loss,
                    best_loss_epoch,
                    model_name,
                )

        adjust_learning_rate(scheduler, valid_loss, params)

        if world_rank == 0:
            logger.info(
                f"Epoch [{epoch + 1}/{params.max_epochs}], "
                f"Train Loss: {train_loss:.4f}, "
                f"Valid Loss: {valid_loss:.4f}, "
                f"Best loss at Epoch: {best_loss_epoch + 1}"
            )
            train_losses = np.append(
                train_losses, train_loss)
            valid_losses = np.append(
                valid_losses, valid_loss)
            np.save(train_loss_file, train_losses)
            np.save(valid_loss_file, valid_losses)

        if epoch - best_loss_epoch > params.patience:
            if is_first_rank:
                logger.info("Early stopping triggered.")
            break
    if world_rank == 0:
        print(
            "=" * 10, f"\n dp cost {time.perf_counter() - t1:.2f} \n", "=" * 10)


def save_checkpoint(
    model, optimizer, scheduler, best_valid_loss, best_loss_epoch, model_name
):
    model_to_save = model.module if hasattr(
        model, "module") else model
    state = {
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_valid_loss": best_valid_loss,
        "best_loss_epoch": best_loss_epoch,
    }
    torch.save(state, model_name)


def get_scheduler(optimizer, params):
    if params.scheduler == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.2, patience=5, mode="min"
        )
    elif params.scheduler == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.T_max)
    else:
        return None


def adjust_learning_rate(scheduler, valid_loss, params):
    if params.scheduler == "ReduceLROnPlateau":
        scheduler.step(valid_loss)
    elif params.scheduler == "CosineAnnealingLR":
        scheduler.step()


if __name__ == "__main__":
    current_path = os.getcwd()
    sys.path.append(current_path)
    main()
