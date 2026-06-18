import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel
from onescience.models.fourcastnet import FourCastNet
from onescience.utils.YParams import YParams
from onescience.utils.fcn.darcy_loss_ocean import LpLoss as loss_function

from apex import optimizers
from dataloader import OceanDatapipe, get_input_channels, get_output_channels, load_or_create_mask, resolve_path


ROOT_DIR = Path(__file__).resolve().parent


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger()

    ## Model config init
    config_file_path = ROOT_DIR / "conf" / "config.yaml"
    cfg = YParams(config_file_path, "model")

    ## Distributed config init
    cfg.world_size = 1
    if "WORLD_SIZE" in os.environ:
        cfg.world_size = int(os.environ["WORLD_SIZE"])
    world_rank = 0
    local_rank = 0
    if cfg.world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ["LOCAL_RANK"])
        world_rank = dist.get_rank()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    ## DataLoader init
    cfg_data = YParams(config_file_path, "datapipe")
    datapipe = OceanDatapipe(
        dataset_cfg=cfg_data.dataset,
        dataloader_cfg=cfg_data.dataloader,
        used_years=cfg_data.dataset.train_time,
        distributed=dist.is_initialized(),
    )
    train_dataloader, train_sampler = datapipe.get_dataloader("train")
    datapipe = OceanDatapipe(
        dataset_cfg=cfg_data.dataset,
        dataloader_cfg=cfg_data.dataloader,
        used_years=cfg_data.dataset.val_time,
        distributed=dist.is_initialized(),
    )
    val_dataloader, val_sampler = datapipe.get_dataloader("valid")

    # Model init
    model = FourCastNet(
        img_size=tuple(cfg_data.dataset.img_size),
        patch_size=tuple(cfg.patch_size),
        in_chans=len(get_input_channels(cfg_data.dataset)),
        out_chans=len(get_output_channels(cfg_data.dataset)),
        num_blocks=cfg.num_blocks,
    ).to(device)
    optimizer = optimizers.FusedAdam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=5, mode="min")

    mask = load_or_create_mask(cfg_data.dataset)
    loss_obj = loss_function(torch.from_numpy(mask).to(device))

    ## Train process init
    checkpoint_dir = resolve_path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_loss_file = checkpoint_dir / "trloss.npy"
    valid_loss_file = checkpoint_dir / "valoss.npy"
    best_valid_loss = 1.0e6
    best_loss_epoch = 0
    train_losses = np.empty((0,), dtype=np.float32)
    valid_losses = np.empty((0,), dtype=np.float32)

    ## Get model params count
    if cfg.world_size == 1:
        total_params = sum(p.numel() for p in model.parameters())
        print("\n\n")
        print("-" * 50)
        print(f"📂 now params is {total_params}, {total_params / 1e6:.2f}M, {total_params / 1e9:.2f}B")
        print("-" * 50, "\n")

    ## Load model weight if there exist well-trained model
    backup_ckpt = checkpoint_dir / "model_bak.pth"
    if backup_ckpt.exists():
        if world_rank == 0:
            print("\n\n")
            print("-" * 50)
            print("✅ There has a model weight, load and continue training...")
            print(f"If you want to train a new model, ensure there is no *.pth file in {checkpoint_dir}")
            print("-" * 50, "\n")
        ckpt = torch.load(backup_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_valid_loss = ckpt["best_valid_loss"]
        best_loss_epoch = ckpt["best_loss_epoch"]
        if train_loss_file.exists():
            train_losses = np.load(train_loss_file)
        if valid_loss_file.exists():
            valid_losses = np.load(valid_loss_file)

    ## Distributed model
    if cfg.world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    world_rank == 0 and logger.info("start training ...")

    for epoch in range(cfg.max_epoch):
        if dist.is_initialized():
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
        model.train()
        train_loss = 0
        start_time = time.time()
        for j, data in enumerate(train_dataloader):
            invar = data[0].to(device=device, dtype=torch.float32)
            outvar = data[1].to(device=device, dtype=torch.float32)
            outvar_pred = model(invar)
            loss = loss_obj(outvar_pred, outvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if world_rank == 0:
                logger.info(
                    f"Train: Epoch {epoch}-{j+1}/{len(train_dataloader)} "
                    f"[cost {int((time.time()-start_time) // 60):02}:{int((time.time()-start_time) % 60):02}] "
                    f"[{(time.time()-start_time)/(j+1): .02f}s/{cfg_data.dataloader.batch_size}batch] "
                    f"loss:{train_loss / (j+1): .04f}"
                )

        train_loss /= len(train_dataloader)

        model.eval()
        valid_loss = 0
        with torch.no_grad():
            start_time = time.time()
            for j, data in enumerate(val_dataloader):
                invar = data[0].to(device=device, dtype=torch.float32)
                outvar = data[1].to(device=device, dtype=torch.float32)
                outvar_pred = model(invar)
                loss = loss_obj(outvar_pred, outvar)

                if cfg.world_size > 1:
                    loss_tensor = loss.detach().to(device)
                    dist.all_reduce(loss_tensor)
                    loss = loss_tensor.item() / cfg.world_size
                    valid_loss += loss
                else:
                    valid_loss += loss.item()
                if world_rank == 0:
                    logger.info(
                        f"Valid: Epoch {epoch}-{j+1}/{len(val_dataloader)} "
                        f"[cost {int((time.time()-start_time) // 60):02}:{int((time.time()-start_time) % 60):02}] "
                        f"[{(time.time()-start_time)/(j+1): .02f}s/{cfg_data.dataloader.batch_size}batch] "
                        f"loss:{valid_loss / (j+1): .04f}"
                    )

        valid_loss /= len(val_dataloader)
        is_save_ckp = False
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_loss_epoch = epoch
            world_rank == 0 and save_checkpoint(model, optimizer, scheduler, best_valid_loss, best_loss_epoch, checkpoint_dir)
            is_save_ckp = True
        scheduler.step(valid_loss)

        if world_rank == 0:
            logger.info(
                f"Epoch [{epoch + 1}/{cfg.max_epoch}], "
                f"Train Loss: {train_loss:.4f}, "
                f"Valid Loss: {valid_loss:.4f}, "
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
    model_path = Path(model_path)
    latest_ckpt = model_path / "model.pth"
    backup_ckpt = model_path / "model_bak.pth"
    torch.save(state, latest_ckpt)
    os.replace(latest_ckpt, backup_ckpt)


if __name__ == "__main__":
    main()
