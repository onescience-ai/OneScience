import logging
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from apex import optimizers
from torch.nn.parallel import DistributedDataParallel

from onescience.datapipes.climate import ERA5HDF5Datapipe
from onescience.memory.checkpoint import replace_function
from onescience.models.pangu import Pangu
from onescience.utils.fcn.YParams import YParams


def loss_func(x, y):
    return torch.nn.functional.l1_loss(x, y)


def main():

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger()

    config_file_path = os.path.join(
        current_path, "conf/config.yaml")
    cfg = YParams(config_file_path, "pangu")
    cfg.world_size = 1
    if "WORLD_SIZE" in os.environ:
        cfg.world_size = int(os.environ["WORLD_SIZE"])
    world_rank = 0
    local_rank = 0

    if cfg.world_size > 1:
        dist.init_process_group(
            backend="nccl", init_method="env://")
        local_rank = int(os.environ["LOCAL_RANK"])
        world_rank = dist.get_rank()

    land_mask = torch.from_numpy(
        np.load(os.path.join(cfg.mask_dir, "land_mask.npy")).astype(
            np.float32)
    )
    soil_type = torch.from_numpy(
        np.load(os.path.join(cfg.mask_dir, "soil_type.npy")).astype(
            np.float32)
    )
    topography = torch.from_numpy(
        np.load(os.path.join(cfg.mask_dir, "topography.npy")).astype(
            np.float32)
    )
    surface_mask = torch.stack(
        [land_mask, soil_type, topography], dim=0).to(local_rank)
    surface_mask = surface_mask.unsqueeze(
        0).repeat(cfg.batch_size, 1, 1, 1)

    train_dataset = ERA5HDF5Datapipe(
        params=cfg, distributed=dist.is_initialized())
    train_dataloader, train_sampler = train_dataset.train_dataloader()
    world_rank == 0 and logger.info(
        f"Loaded train_dataloader of size {len(train_dataloader)}"
    )

    val_dataset = ERA5HDF5Datapipe(
        params=cfg, distributed=dist.is_initialized())
    val_dataloader, val_sampler = val_dataset.val_dataloader()
    world_rank == 0 and logger.info(
        f"Loaded val_dataloader of size {len(val_dataloader)}"
    )

    pangu_model = Pangu(
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        window_size=cfg.window_size,
    ).to(local_rank)

    if cfg.world_size > 1:
        pangu_model = DistributedDataParallel(
            pangu_model, device_ids=[
                local_rank], output_device=local_rank
        )

    optimizer = optimizers.FusedAdam(
        pangu_model.parameters(), betas=(0.9, 0.999), lr=5e-4, weight_decay=3e-6
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    train_loss_file = f"{cfg.checkpoint_dir}/trloss.npy"
    valid_loss_file = f"{cfg.checkpoint_dir}/valoss.npy"

    world_rank == 0 and logger.info(f"start training ...")

    best_valid_loss = 1.0e6
    best_loss_epoch = 0
    train_losses = np.empty((0,), dtype=np.float32)
    valid_losses = np.empty((0,), dtype=np.float32)

    for epoch in range(cfg.max_epoch):

        epoch_start_time = time.time()  # 记录epoch开始时间

        if dist.is_initialized():
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)

        pangu_model.train()
        train_loss = 0
        batch_start_time = time.time()
        for j, data in enumerate(train_dataloader):
            invar = data[0]
            outvar = data[1]
            invar_surface = invar[:, :4, :, :].to(
                local_rank, dtype=torch.float32)
            invar_upper_air = invar[:, 4:, :, :].to(
                local_rank, dtype=torch.float32)
            invar = torch.concat(
                [invar_surface, surface_mask, invar_upper_air], dim=1)

            tar_surface = outvar[:, :4, :, :].to(
                local_rank, dtype=torch.float32)
            tar_upper_air = outvar[:, 4:, :, :].to(
                local_rank, dtype=torch.float32)

            with replace_function(
                pangu_model,
                ["layer1", "layer2", "layer3", "layer4"],
                cfg.world_size > 1,
            ):
                out_surface, out_upper_air = pangu_model(
                    invar)

            out_upper_air = out_upper_air.reshape(
                tar_upper_air.shape)

            loss1 = loss_func(tar_surface, out_surface)
            loss2 = loss_func(tar_upper_air, out_upper_air)
            loss = loss1 * 0.25 + loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if j % (len(train_dataloader) // 6) == 0 and world_rank == 0:
                batch_time = time.time() - batch_start_time
                logger.info(
                    f"Epoch [{epoch + 1}/{cfg.max_epoch}], Train MiniBatch {j}/{len(train_dataloader)} done, "
                    f"MiniBatch Time: {batch_time:.2f}s, Current Loss: {loss.item():.4f}"
                )
                batch_start_time = time.time()

        train_loss /= len(train_dataloader)

        pangu_model.eval()
        valid_loss = 0
        val_batch_start_time = time.time()
        with torch.no_grad():
            for j, data in enumerate(val_dataloader):
                invar = data[0]
                outvar = data[1]
                invar_surface = invar[:, :4, :, :].to(
                    local_rank, dtype=torch.float32)
                invar_upper_air = invar[:, 4:, :, :].to(
                    local_rank, dtype=torch.float32)
                invar = torch.concat(
                    [invar_surface, surface_mask,
                        invar_upper_air], dim=1
                )

                tar_surface = outvar[:, :4, :, :].to(
                    local_rank, dtype=torch.float32)
                tar_upper_air = outvar[:, 4:, :, :].to(
                    local_rank, dtype=torch.float32)

                out_surface, out_upper_air = pangu_model(
                    invar)
                out_upper_air = out_upper_air.reshape(
                    tar_upper_air.shape)

                loss1 = loss_func(
                    tar_surface, out_surface).item()
                loss2 = loss_func(
                    tar_upper_air, out_upper_air).item()
                loss = loss1 * 0.25 + loss2

                if cfg.world_size > 1:
                    loss_tensor = torch.tensor(
                        loss, device=local_rank)
                    dist.all_reduce(loss_tensor)
                    loss = loss_tensor.item() / cfg.world_size
                valid_loss += loss

                if j % (len(val_dataloader) // 3) == 0 and world_rank == 0:
                    val_batch_time = time.time() - val_batch_start_time
                    logger.info(
                        f"Epoch [{epoch + 1}/{cfg.max_epoch}], Val MiniBatch {j}/{len(val_dataloader)} done, "
                        f"MiniBatch Time: {val_batch_time:.2f}s, Current Loss: {loss:.4f}"
                    )
                    val_batch_start_time = time.time()
        valid_loss /= len(val_dataloader)
        is_save_ckp = False
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_loss_epoch = epoch
            world_rank == 0 and save_checkpoint(
                pangu_model,
                optimizer,
                scheduler,
                best_valid_loss,
                best_loss_epoch,
                cfg.checkpoint_dir,
            )
            is_save_ckp = True

        scheduler.step()

        epoch_time = time.time() - epoch_start_time  # 计算epoch耗时

        if world_rank == 0:
            logger.info(
                f"Epoch [{epoch + 1}/{cfg.max_epoch}] finished in {epoch_time:.2f}s, "
                f"Train Loss: {train_loss:.4f}, "
                f"Valid Loss: {valid_loss:.4f}, "
                f"Best loss at Epoch: {best_loss_epoch + 1}"
                + (", saving checkpoint" if is_save_ckp else "")
            )
            train_losses = np.append(
                train_losses, train_loss)
            valid_losses = np.append(
                valid_losses, valid_loss)

            np.save(train_loss_file, train_losses)
            np.save(valid_loss_file, valid_losses)
        if epoch - best_loss_epoch > cfg.patience:
            print(
                f"Loss has not decrease in {cfg.patience} epochs, stopping training..."
            )
            exit()


def save_checkpoint(
    model, optimizer, scheduler, best_valid_loss, best_loss_epoch, model_path
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
    torch.save(state, f"{model_path}/pangu_weather.pth")


if __name__ == "__main__":
    current_path = os.getcwd()
    sys.path.append(current_path)
    main()
