import torch
import os
import sys
import numpy as np
import torch.distributed as dist
import logging
import time

from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, LambdaLR
from onescience.datapipes import ERA5Datapipe
from onescience.utils.YParams import YParams
from onescience.utils.graphcast.data_utils import StaticData
from onescience.utils.graphcast.graph_utils import deg2rad
from onescience.models.graphcast.graph_cast_net import GraphCastNet
from onescience.utils.graphcast.loss import GraphCastLossFunction
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

    ## DataLoader init
    cfg_data = YParams(config_file_path, "datapipe")

    input_dim_grid_nodes = (len(cfg_data.dataset.channels) + cfg.use_cos_zenith + 4 * cfg.use_time_of_year_index) * (cfg.num_history + 1) + cfg.num_channels_static
    
    model = GraphCastNet(mesh_level=cfg.mesh_level,
                         multimesh=cfg.multimesh,
                         input_res=tuple(cfg_data.dataset.img_size),
                         input_dim_grid_nodes=input_dim_grid_nodes,
                         input_dim_mesh_nodes=3,
                         input_dim_edges=4,
                         output_dim_grid_nodes=len(cfg_data.dataset.channels),
                         processor_type=cfg.processor_type,
                         khop_neighbors=cfg.khop_neighbors,
                         num_attention_heads=cfg.num_attention_heads,
                         processor_layers=cfg.processor_layers,
                         hidden_dim=cfg.hidden_dim,
                         norm_type=cfg.norm_type,
                         do_concat_trick=cfg.concat_trick,
                         recompute_activation=cfg.recompute_activation,
                         )
    model_dtype = torch.bfloat16 if cfg.full_bf16 else torch.float32
    model.set_checkpoint_encoder(cfg.checkpoint_encoder)
    model.set_checkpoint_decoder(cfg.checkpoint_decoder)
    model = model.to(dtype=model_dtype).to(local_rank)
    if hasattr(model, "module"):
        latitudes = model.module.latitudes
        longitudes = model.module.longitudes
        lat_lon_grid = model.module.lat_lon_grid
    else:
        latitudes = model.latitudes
        longitudes = model.longitudes
        lat_lon_grid = model.lat_lon_grid
    static_data = StaticData(cfg_data.dataset.static_dir, latitudes, longitudes).get().to(device=local_rank)
    channels_list = [i for i in range(len(cfg_data.dataset.channels))]
    area = torch.abs(torch.cos(deg2rad(lat_lon_grid[:, :, 0])))
    area /= torch.mean(area)
    area = area.to(dtype=torch.bfloat16 if cfg.full_bf16 else torch.float32).to(device=local_rank)
    criterion = GraphCastLossFunction(area, channels_list, cfg_data.dataset.dataset_metadata_path, cfg_data.dataset.time_diff_std_path)
    optimizer = optimizers.FusedAdam(model.parameters(),
                                     lr=cfg.lr, betas=(0.9, 0.95),
                                     adam_w_mode=True,
                                     weight_decay=0.1)
    scheduler1 = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=cfg.num_iters_step1, )
    scheduler2 = CosineAnnealingLR(optimizer, T_max=cfg.num_iters_step2, eta_min=0.0)
    scheduler3 = LambdaLR(optimizer, lr_lambda=lambda epoch: (cfg.lr_step3 / cfg.lr))
    scheduler = SequentialLR(optimizer,
                             schedulers=[scheduler1, scheduler2, scheduler3],
                             milestones=[cfg.num_iters_step1, cfg.num_iters_step1 + cfg.num_iters_step2])
    
    ## Train process init
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    train_loss_file = f"{cfg.checkpoint_dir}/ft_trloss.npy"
    best_valid_loss = 1.0e6
    best_loss_epoch = 0
    train_losses = np.empty((0,), dtype=np.float32)

    ## Get model params count
    if cfg.world_size == 1:
        total_params = sum(p.numel() for p in model.parameters())
        print("\n\n")
        print("-" * 50)
        print(f"📂 now params is {total_params}, {total_params / 1e6:.2f}M, {total_params / 1e9:.2f}B")
        print("-" * 50, "\n")

    ## Load model weight if there exist well-trained model 
    if not os.path.exists(f"{cfg.checkpoint_dir}/model_bak.pth"):
        print('⚠️ ⚠️ Please train the model first...')
        exit()
    
    if os.path.exists(f"{cfg.checkpoint_dir}/model_finetune_bak.pth"):
        if world_rank == 0:
            print("\n\n")
            print("-" * 50)
            print(f"✅ There has a finetune-model weight, load and continue training...")
            print(f'If you want to finetune a new model, ensure there is no model_finetune_bak.pth file in {cfg.checkpoint_dir}')
            print("-" * 50, "\n")
        ckpt = torch.load(f"{cfg.checkpoint_dir}/model_bak.pth", map_location=f'cuda:{local_rank}', weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_valid_loss = ckpt["best_valid_loss"]
        best_loss_epoch = ckpt["best_loss_epoch"]
        train_losses = np.load(train_loss_file)
    else:
        if world_rank == 0:
            print("\n\n")
            print("-" * 50)
            print(f"✅ Load well-trained model and continue to finetune...")
            print(f'If you want to train a new model, ensure there is no *.pth file in {cfg.checkpoint_dir}')
            print("-" * 50, "\n")
        ckpt = torch.load(f"{cfg.checkpoint_dir}/model_bak.pth", map_location=f'cuda:{local_rank}', weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    if cfg.world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    if cfg.force_single_checkpoint_finetune:
        if hasattr(model, "module"):
            model.module.set_checkpoint_model(True)
        else:
            model.set_checkpoint_model(True)
    if cfg.checkpoint_encoder_finetune:
        if hasattr(model, "module"):
            model.module.set_checkpoint_encoder(True)
        else:
            model.set_checkpoint_encoder(True)
    if cfg.checkpoint_processor_finetune:
        if hasattr(model, "module"):
            model.module.set_checkpoint_processor(cfg.segments)
        else:
            model.set_checkpoint_encoder(True)
    if cfg.checkpoint_decoder_finetune:
        if hasattr(model, "module"):
            model.module.set_checkpoint_decoder(True)
        else:
            model.set_checkpoint_encoder(True)

    world_rank == 0 and logger.info(f"start finetuning ...")


    num_rollout_steps = 2
    world_rank == 0 and logger.info(f"Switching to {num_rollout_steps}-step rollout!")
    datapipe = ERA5Datapipe(params=cfg_data, distributed=dist.is_initialized(), output_steps=num_rollout_steps)
    train_dataloader, train_sampler = datapipe.train_dataloader()
    val_dataloader, val_sampler = datapipe.val_dataloader()

    for epoch in range(cfg.num_iters_step3):
        if epoch % cfg.step_change_freq == 0:
            num_rollout_steps = epoch // cfg.step_change_freq + 2
            world_rank == 0 and logger.info(f"Switching to {num_rollout_steps}-step rollout!")
            datapipe = ERA5Datapipe(params=cfg_data, distributed=dist.is_initialized(), output_steps=num_rollout_steps)
            train_dataloader, train_sampler = datapipe.train_dataloader()

        if dist.is_initialized():
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
        train_loss = 0
        start_time = time.time()
        for j, data in enumerate(train_dataloader):
            invar = data[0].to(device=local_rank)
            
            outvar = data[1].to(device=local_rank)
            cos_zenith = data[2].to(device=local_rank)
            in_idx = data[3].item()
            cos_zenith = torch.squeeze(cos_zenith, dim=2)
            cos_zenith = torch.clamp(cos_zenith, min=0.0) - 1.0 / torch.pi
            outvar = outvar.to(dtype=model_dtype)
            for t in range(outvar.shape[1]):
                day_of_year, time_of_day = divmod(in_idx + t * cfg.dt, 24 // cfg.dt)
                normalized_day_of_year = torch.tensor((day_of_year / 365) * (np.pi / 2), dtype=torch.float32, device=local_rank)
                normalized_time_of_day = torch.tensor((time_of_day / (24 - cfg.dt)) * (np.pi / 2), dtype=torch.float32, device=local_rank)
                sin_day_of_year = torch.sin(normalized_day_of_year).expand(1, 1, 721, 1440)
                cos_day_of_year = torch.cos(normalized_day_of_year).expand(1, 1, 721, 1440)
                sin_time_of_day = torch.sin(normalized_time_of_day).expand(1, 1, 721, 1440)
                cos_time_of_day = torch.cos(normalized_time_of_day).expand(1, 1, 721, 1440)
                invar = torch.concat((invar, cos_zenith[:, t:t + 1, :, :], static_data, sin_day_of_year, cos_day_of_year, sin_time_of_day, cos_time_of_day), dim=1)
                invar = invar.to(dtype=model_dtype)
                if t < outvar.shape[1] - 1:
                    with torch.no_grad():
                        outpred = model(invar)
                    invar = outpred.detach()
                else:
                    outpred = model(invar)
                    loss = criterion(outpred, outvar[:, t])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            torch.cuda.nvtx.range_pop()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            if world_rank == 0:
                logger.info(f'Train: Epoch {epoch} {j+1}/{len(train_dataloader)} '
                            f'[cost {int((time.time()-start_time) // 60):02}:{int((time.time()-start_time) % 60):02}] '
                            f'[{(time.time()-start_time)/(j+1): .02f}s/{cfg_data.dataloader.batch_size}batch] '
                            f'loss:{train_loss / (j+1): .04f}')

            if (j + 1) % cfg.val_freq == 0:
                model.eval()
                valid_loss = 0.0
                with torch.no_grad():
                    start_time = time.time()
                    for k, data in enumerate(val_dataloader):
                        invar = data[0].to(device=local_rank)
                        outvar = data[1].to(device=local_rank)
                        cos_zenith = data[2].to(device=local_rank)
                        in_idx = data[3].item()

                        cos_zenith = torch.squeeze(cos_zenith, dim=2)
                        cos_zenith = torch.clamp(cos_zenith, min=0.0) - 1.0 / torch.pi  # [b, 2, h, w]
                        outvar = outvar.to(dtype=model_dtype)
                        loss = 0.0
                        for t in range(outvar.shape[1]):
                            day_of_year, time_of_day = divmod(in_idx + t * cfg.dt, 24 // cfg.dt)
                            normalized_day_of_year = torch.tensor((day_of_year / 365) * (np.pi / 2), dtype=torch.float32, device=local_rank)
                            normalized_time_of_day = torch.tensor((time_of_day / (24 - cfg.dt)) * (np.pi / 2), dtype=torch.float32, device=local_rank)
                            sin_day_of_year = torch.sin(normalized_day_of_year).expand(1, 1, 721, 1440)
                            cos_day_of_year = torch.cos(normalized_day_of_year).expand(1, 1, 721, 1440)
                            sin_time_of_day = torch.sin(normalized_time_of_day).expand(1, 1, 721, 1440)
                            cos_time_of_day = torch.cos(normalized_time_of_day).expand(1, 1, 721, 1440)
                            invar = torch.concat((invar, cos_zenith[:, t:t + 1, :, :], static_data, sin_day_of_year, cos_day_of_year, sin_time_of_day, cos_time_of_day), dim=1)
                            invar = invar.to(dtype=model_dtype)
                            outpred = model(invar)
                            invar = outpred
                            loss += criterion(outpred, outvar[:, t])

                        loss /= outvar.shape[1]
                        if cfg.world_size > 1:
                            loss_tensor = loss.detach().to(local_rank)  # torch.tensor(loss, device=local_rank)
                            dist.all_reduce(loss_tensor)
                            loss = loss_tensor.item() / cfg.world_size
                            valid_loss += loss
                        else:
                            valid_loss += loss.item()

                        if world_rank == 0:
                            logger.info(f'Valid: Epoch {epoch} {k+1}/{len(val_dataloader)} '
                                    f'[cost {int((time.time()-start_time) // 60):02}:{int((time.time()-start_time) % 60):02}] '
                                    f'[{(time.time()-start_time)/(k+1): .02f}s/{cfg_data.dataloader.batch_size}batch] '
                                    f'loss:{valid_loss / (k+1): .04f}')
                                    
                        break

                    valid_loss /= len(val_dataloader)
                    is_save_ckp = False
                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        best_loss_epoch = epoch
                        world_rank == 0 and save_checkpoint(model, optimizer, scheduler, best_valid_loss, best_loss_epoch, cfg.checkpoint_dir)
                        is_save_ckp = True
                train_loss /= (j+1)
                if world_rank == 0:
                    logger.info(f"Epoch [{epoch + 1}/{cfg.max_epoch}], "
                                f"Train Loss: {train_loss:.4f}, "
                                f"Valid Loss: {valid_loss:.4f}, "
                                f"Best loss at Epoch: {best_loss_epoch + 1}"
                                + (", saving checkpoint" if is_save_ckp else "")
                                )
                    train_losses = np.append(train_losses, train_loss)
                    np.save(train_loss_file, train_losses)
                    exit()

    print(f'Graphcast has been well-trained, next step is fine-tune')


def save_checkpoint(model, optimizer, scheduler, best_valid_loss, best_loss_epoch, model_path):
    model_to_save = model.module if hasattr(model, "module") else model
    state = {"model_state_dict": model_to_save.state_dict(),
             "optimizer_state_dict": optimizer.state_dict(),
             "scheduler_state_dict": scheduler.state_dict(),
             "best_valid_loss": best_valid_loss,
             "best_loss_epoch": best_loss_epoch,
            }
    torch.save(state, f"{model_path}/model_finetune.pth")
    ### the weight file saving may interrupted due to DCU queue limit, get a backup to ensure there at least has one model 
    os.system(f"mv {model_path}/model_finetune.pth {model_path}/model_finetune_bak.pth")


if __name__ == "__main__":
    current_path = os.getcwd()
    sys.path.append(current_path)
    main()