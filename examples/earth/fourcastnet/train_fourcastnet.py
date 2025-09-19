import torch
import os
import sys
import numpy as np
import torch.distributed as dist
import logging
import time

from torch.nn.parallel import DistributedDataParallel
from onescience.models.afno.afnonet import AFNONet
from onescience.datapipes.climate import ERA5HDF5Datapipe
from onescience.utils.fcn.YParams import YParams
from onescience.utils.fcn.darcy_loss import LpLoss

from apex import optimizers


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger()

    config_file_path = os.path.join(current_path, "conf/config.yaml")
    cfg = YParams(config_file_path, "fourcastnet")
    cfg['N_in_channels'] = len(cfg.channels)
    cfg['N_out_channels'] = len(cfg.channels)
    cfg.world_size = 1
    if "WORLD_SIZE" in os.environ:
        cfg.world_size = int(os.environ["WORLD_SIZE"])
    world_rank = 0
    local_rank = 0

    if cfg.world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ["LOCAL_RANK"])
        world_rank = dist.get_rank()

    train_dataset = ERA5HDF5Datapipe(params=cfg, distributed=dist.is_initialized())
    train_dataloader, train_sampler = train_dataset.train_dataloader()
    world_rank == 0 and logger.info(f"Loaded train_dataloader of size {len(train_dataloader)}")

    val_dataset = ERA5HDF5Datapipe(params=cfg, distributed=dist.is_initialized())
    val_dataloader, val_sampler = val_dataset.val_dataloader()
    world_rank == 0 and logger.info(f"Loaded val_dataloader of size {len(val_dataloader)}")

    fourcastnet_model = AFNONet(cfg).to(local_rank)

    if cfg.world_size > 1:
        fourcastnet_model = DistributedDataParallel(fourcastnet_model, device_ids=[local_rank], output_device=local_rank)

    optimizer = optimizers.FusedAdam(fourcastnet_model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=5, mode='min')
    loss_obj = LpLoss()

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    train_loss_file = f"{cfg.checkpoint_dir}/trloss.npy"
    valid_loss_file = f"{cfg.checkpoint_dir}/valoss.npy"

    world_rank == 0 and logger.info(f"start training ...")

    best_valid_loss = 1.0e6
    best_loss_epoch = 0
    train_losses = np.empty((0,), dtype=np.float32)
    valid_losses = np.empty((0,), dtype=np.float32)

    print_length = 1 # len(train_dataloader) // 64

    for epoch in range(cfg.max_epoch):

        epoch_start_time = time.time()  # 记录epoch开始时间

        if dist.is_initialized():
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
        fourcastnet_model.train()
        train_loss = 0
        batch_start_time = time.time()
        for j, data in enumerate(train_dataloader):
            invar = data[0].to(local_rank, dtype=torch.float32)
            outvar = data[1].to(local_rank, dtype=torch.float32)

            invar = invar[:, :, :-1, :]
            outvar = outvar[:, :, :-1, :]

            outvar_pred = fourcastnet_model(invar)

            loss = loss_obj(outvar, outvar_pred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if world_rank == 0 and j % print_length == 0 :
                batch_time = time.time() - batch_start_time
                logger.info(f"Epoch [{epoch + 1}/{cfg.max_epoch}], Train MiniBatch {j}/{len(train_dataloader)} done, "
                            f"Each MiniBatch Cost: {batch_time / print_length:.2f}s, Current Loss: {loss.item():.4f}")
                batch_start_time = time.time()

        train_loss /= len(train_dataloader)

        fourcastnet_model.eval()
        valid_loss = 0
        val_batch_time = time.time()
        with torch.no_grad():
            for j, data in enumerate(val_dataloader):
                invar = data[0].to(local_rank, dtype=torch.float32)
                outvar = data[1].to(local_rank, dtype=torch.float32)

                invar = invar[:, :, :-1, :]
                outvar = outvar[:, :, :-1, :]

                outvar_pred = fourcastnet_model(invar)

                loss = loss_obj(outvar, outvar_pred)

                if cfg.world_size > 1:
                    loss_tensor = loss.detach().to(local_rank) # torch.tensor(loss, device=local_rank)
                    dist.all_reduce(loss_tensor)
                    loss = loss_tensor.item() / cfg.world_size
                    valid_loss += loss
                else:
                    valid_loss += loss.item()

                if world_rank == 0 and j % print_length == 0 :
                    val_batch_time = time.time() - val_batch_time
                    logger.info(f"Epoch [{epoch + 1}/{cfg.max_epoch}], Val MiniBatch {j}/{len(val_dataloader)} done, "
                                f"Each MiniBatch Time: {val_batch_time/print_length:.2f}s, Current Loss: {loss:.4f}")
                    val_batch_time = time.time()

        valid_loss /= len(val_dataloader)
        is_save_ckp = False
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_loss_epoch = epoch
            world_rank == 0 and save_checkpoint(fourcastnet_model, optimizer, scheduler, best_valid_loss, best_loss_epoch, cfg.checkpoint_dir)
            is_save_ckp = True
        scheduler.step(valid_loss)
        epoch_time = time.time() - epoch_start_time  # 计算epoch耗时

        if world_rank == 0:
            logger.info(
                f"Epoch [{epoch + 1}/{cfg.max_epoch}] finished in {epoch_time:.2f}s, "
                f"Train Loss: {train_loss:.4f}, "
                f"Valid Loss: {valid_loss:.4f}, "
                f"This Epoch cost {time.time() - epoch_start_time: .2f}, "
                f"Best loss at Epoch: {best_loss_epoch + 1}"
                + (", saving checkpoint" if is_save_ckp else "")
            )
            train_losses = np.append(train_losses, train_loss)
            valid_losses = np.append(valid_losses, valid_loss)

            np.save(train_loss_file, train_losses)
            np.save(valid_loss_file, valid_losses)
        if epoch - best_loss_epoch > cfg.patience:
            print(f"Loss has not decrease in {cfg.patience} epochs, stopping training...")
            exit()


def save_checkpoint(model, optimizer, scheduler, best_valid_loss, best_loss_epoch, model_path):
    model_to_save = model.module if hasattr(model, "module") else model
    state = {
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_valid_loss": best_valid_loss,
        "best_loss_epoch": best_loss_epoch,
    }
    torch.save(state, f"{model_path}/fourcastnet.pth")


if __name__ == "__main__":
    current_path = os.getcwd()
    sys.path.append(current_path)
    main()
