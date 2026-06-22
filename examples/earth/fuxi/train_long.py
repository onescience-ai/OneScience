import torch
import os
import sys
import numpy as np
import torch.distributed as dist
import logging
import time

from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel
from onescience.models.fuxi import Fuxi
from data_loader import ERA5Datapipe
from onescience.utils.YParams import YParams
from onescience.metrics.climate.loss import LatitudeWeightedLoss
from onescience.memory.checkpoint import replace_function

from apex import optimizers


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger()

    ## Model config init
    config_file_path = os.path.join(current_path, "conf/config.yaml")
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
    if not os.path.exists(f"{cfg.checkpoint_dir}/model_medium_bak.pth"):
        if world_rank == 0:
            print(f'❌❌The Fuxi medium model must be trained before this model.')
        exit()
    ## DataLoader init
    cfg_data = YParams(config_file_path, "datapipe")
    datapipe = ERA5Datapipe(
        dataset_dir=cfg_data.dataset.data_dir,
        used_variables=cfg_data.dataset.channels,
        used_years=cfg_data.dataset.train_time,
        pattern='long',
        distributed=dist.is_initialized(),
        output_steps=2,
        input_steps=2,
    )
    train_dataloader, train_sampler = datapipe.get_dataloader("train")
    datapipe = ERA5Datapipe(
        dataset_dir=cfg_data.dataset.data_dir,
        used_variables=cfg_data.dataset.channels,
        used_years=cfg_data.dataset.val_time,
        pattern='long',
        distributed=dist.is_initialized(),
        output_steps=2,
        input_steps=2,
    )
    val_dataloader, val_sampler = datapipe.get_dataloader("valid")

    ## Model init
    model = Fuxi(img_size=cfg_data.dataset.img_size, 
                 patch_size=cfg.patch_size, 
                 in_chans=len(cfg_data.dataset.channels),
                 out_chans=len(cfg_data.dataset.channels),
                 embed_dim=cfg.embed_dim, 
                 num_groups=cfg.num_groups, 
                 num_heads=cfg.num_heads, 
                 window_size=cfg.window_size
                 ).to(local_rank)
    optimizer = optimizers.FusedAdam(model.parameters(), lr=cfg.train_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=5, mode="min")
    loss_obj = LatitudeWeightedLoss(loss_type="l1", normalize=True).to(local_rank)

    ## Train process init
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    train_loss_file = f"{cfg.checkpoint_dir}/tr_long_loss.npy"
    valid_loss_file = f"{cfg.checkpoint_dir}/va_long_loss.npy"
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
    
    if not os.path.exists(f"{cfg.checkpoint_dir}/model_medium_bak.pth"):
        print('⚠️ ⚠️ Please train to get medium model first...')
        exit()
    
    if os.path.exists(f"{cfg.checkpoint_dir}/model_long_bak.pth"):
        if world_rank == 0:
            print("\n\n")
            print("-" * 50)
            print(f"✅ There has a long-pattern model weight, load and continue training...")
            print(f'If you want to finetune a new model, ensure there is no model_short_bak.pth file in {cfg.checkpoint_dir}')
            print("-" * 50, "\n")
        ckpt = torch.load(f"{cfg.checkpoint_dir}/model_long_bak.pth", map_location=f'cuda:{local_rank}', weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_valid_loss = ckpt["best_valid_loss"]
        best_loss_epoch = ckpt["best_loss_epoch"]
        train_losses = np.load(f"{cfg.checkpoint_dir}/tr_long_loss.npy")
        valid_losses = np.load(f"{cfg.checkpoint_dir}/va_long_loss.npy")
    else:
        if world_rank == 0:
            print("\n\n")
            print("-" * 50)
            print(f"✅ Load medium model and continue to finetune...")
            print("-" * 50, "\n")
        ckpt = torch.load(f"{cfg.checkpoint_dir}/model_medium_bak.pth", map_location=f'cuda:{local_rank}', weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

     ## Distributed model
    if cfg.world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    world_rank == 0 and logger.info(f"start training ...")

    for epoch in range(cfg.finetune_step):
        if epoch % cfg.step_change_freq == 0:
            num_rollout_steps = epoch // cfg.step_change_freq + 2
            if num_rollout_steps > 12: # Paper: 2~12 curriculum training schedule, then skip to 20.
                num_rollout_steps = cfg.long_num_steps - cfg.medium_num_steps
            world_rank == 0 and logger.info(f"⚠️ ⚠️ Switching to {num_rollout_steps}-step rollout!")
            datapipe = ERA5Datapipe(
                dataset_dir=cfg_data.dataset.data_dir,
                used_variables=cfg_data.dataset.channels,
                used_years=cfg_data.dataset.train_time,
                pattern='long',
                distributed=dist.is_initialized(),
                output_steps=num_rollout_steps,
                input_steps=2,
            )
            train_dataloader, train_sampler = datapipe.get_dataloader("train")
            datapipe = ERA5Datapipe(
                dataset_dir=cfg_data.dataset.data_dir,
                used_variables=cfg_data.dataset.channels,
                used_years=cfg_data.dataset.val_time,
                pattern='long',
                distributed=dist.is_initialized(),
                output_steps=num_rollout_steps,
                input_steps=2,
            )
            val_dataloader, val_sampler = datapipe.get_dataloader("valid")
        
        if dist.is_initialized():
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
            
        model.train()
        train_loss = 0
        start_time = time.time()
        for j, data in enumerate(train_dataloader):
            invar = data[0].to(local_rank, dtype=torch.float32) # B, T, C, H, W
            invar = invar.permute(0, 2, 1, 3, 4) # B, C, T, H, W
            outvar = data[1].to(local_rank, dtype=torch.float32)
            for t in range(outvar.shape[1]):
                if t < outvar.shape[1] - 1:
                    with torch.no_grad():
                        outvar_pred = model(invar)
                    # B, 70, 2, 721, 1440
                    invar[:, :, 0] = invar[:, :, -1]
                    invar[:, :, -1] = outvar_pred.detach()
                else:
                    with replace_function(model, ["cube_embedding", "u_transformer"], cfg.world_size > 1):
                        outvar_pred = model(invar)
                    loss = loss_obj(outvar_pred, outvar[:, t])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if world_rank == 0:
                logger.info(f'Train: Epoch {epoch}-{j+1}/{len(train_dataloader)} '
                            f'[cost {int((time.time()-start_time) // 60):02}:{int((time.time()-start_time) % 60):02}] '
                            f'[{(time.time()-start_time)/(j+1): .02f}s/{cfg_data.dataloader.batch_size}batch] '
                            f'loss:{train_loss / (j+1): .04f}')
            
        train_loss /= len(train_dataloader)

        model.eval()
        valid_loss = 0
        with torch.no_grad():
            start_time = time.time()
            for j, data in enumerate(val_dataloader):
                invar = data[0].to(local_rank, dtype=torch.float32) # B, T, C, H, W
                invar = invar.permute(0, 2, 1, 3, 4) # B, C, T, H, W
                outvar = data[1].to(local_rank, dtype=torch.float32)
                for t in range(outvar.shape[1]):
                    outvar_pred = model(invar)
                    # B, 70, 2, 721, 1440
                    invar[:, :, 0] = invar[:, :, -1]
                    invar[:, :, -1] = outvar_pred.detach()
                loss = loss_obj(outvar_pred, outvar[:, -1])

                if cfg.world_size > 1:
                    loss_tensor = loss.detach().to(local_rank)
                    dist.all_reduce(loss_tensor)
                    loss = loss_tensor.item() / cfg.world_size
                    valid_loss += loss
                else:
                    valid_loss += loss.item()
                if world_rank == 0:
                    logger.info(f'Valid: Epoch {epoch}-{j+1}/{len(val_dataloader)} '
                            f'[cost {int((time.time()-start_time) // 60):02}:{int((time.time()-start_time) % 60):02}] '
                            f'[{(time.time()-start_time)/(j+1): .02f}s/{cfg_data.dataloader.batch_size}batch] '
                            f'loss:{valid_loss / (j+1): .04f}')
                
        valid_loss /= len(val_dataloader)
        is_save_ckp = False
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_loss_epoch = epoch
            world_rank == 0 and save_checkpoint(model, optimizer, scheduler, best_valid_loss, best_loss_epoch, cfg.checkpoint_dir)
            is_save_ckp = True

        scheduler.step(valid_loss)

        if world_rank == 0:
            logger.info(f"Epoch [{epoch + 1}/{cfg.max_epoch}], "
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
    state = {"model_state_dict": model_to_save.state_dict(),
             "optimizer_state_dict": optimizer.state_dict(),
             "scheduler_state_dict": scheduler.state_dict(),
             "best_valid_loss": best_valid_loss,
             "best_loss_epoch": best_loss_epoch,
            }
    torch.save(state, f"{model_path}/model_long.pth")
    ### the weight file saving may interrupted due to DCU queue limit, get a backup to ensure there at least has one model 
    os.system(f"mv {model_path}/model_long.pth {model_path}/model_long_bak.pth")


if __name__ == "__main__":
    current_path = os.getcwd()
    sys.path.append(current_path)
    main()