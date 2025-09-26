import logging
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from apex import optimizers
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, SequentialLR

from onescience.datapipes.climate import ERA5HDF5Datapipe
from onescience.launch.utils import save_checkpoint
from onescience.models.graphcast.graph_cast_net import GraphCastNet
from onescience.utils.fcn.YParams import YParams
from onescience.utils.graphcast.data_utils import StaticData
from onescience.utils.graphcast.graph_utils import deg2rad
from onescience.utils.graphcast.loss import GraphCastLossFunction


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger()
    config_file_path = os.path.join(
        current_path, "conf/config.yaml")
    cfg = YParams(config_file_path, "graphcast")

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
    if not torch.cuda.is_bf16_supported():
        cfg.full_bf16 = False

    model_dtype = torch.bfloat16 if cfg.full_bf16 else torch.float32

    train_dataset = ERA5HDF5Datapipe(
        params=cfg, distributed=dist.is_initialized())
    train_dataloader, train_sampler = train_dataset.train_dataloader()
    world_rank == 0 and logger.info(
        f"Loaded train_dataloader of size {len(train_dataloader)}"
    )

    val_dataset = ERA5HDF5Datapipe(
        params=cfg, distributed=dist.is_initialized(), num_steps=cfg.num_val_steps
    )
    val_dataloader, val_sampler = val_dataset.val_dataloader()
    world_rank == 0 and logger.info(
        f"Loaded val_dataloader of size {len(val_dataloader)}"
    )

    input_dim_grid_nodes = (
        len(cfg.channels) + cfg.use_cos_zenith +
        4 * cfg.use_time_of_year_index
    ) * (cfg.num_history + 1) + cfg.num_channels_static
    graphcast_model = GraphCastNet(
        mesh_level=cfg.mesh_level,
        multimesh=cfg.multimesh,
        input_res=tuple(cfg.img_size),
        input_dim_grid_nodes=input_dim_grid_nodes,
        input_dim_mesh_nodes=3,
        input_dim_edges=4,
        output_dim_grid_nodes=len(cfg.channels),
        processor_type=cfg.processor_type,
        khop_neighbors=cfg.khop_neighbors,
        num_attention_heads=cfg.num_attention_heads,
        processor_layers=cfg.processor_layers,
        hidden_dim=cfg.hidden_dim,
        norm_type=cfg.norm_type,
        do_concat_trick=cfg.concat_trick,
        recompute_activation=cfg.recompute_activation,
    )

    graphcast_model.set_checkpoint_encoder(
        cfg.checkpoint_encoder)
    graphcast_model.set_checkpoint_decoder(
        cfg.checkpoint_decoder)
    graphcast_model = graphcast_model.to(
        dtype=model_dtype).to(local_rank)

    world_rank == 0 and logger.info(
        f"Model parameters is {sum(p.numel() for p in graphcast_model.parameters() if p.requires_grad)}"
    )
    if hasattr(graphcast_model, "module"):
        latitudes = graphcast_model.module.latitudes
        longitudes = graphcast_model.module.longitudes
        lat_lon_grid = graphcast_model.module.lat_lon_grid
    else:
        latitudes = graphcast_model.latitudes
        longitudes = graphcast_model.longitudes
        lat_lon_grid = graphcast_model.lat_lon_grid
    static_data = (
        StaticData(cfg.static_dataset_path,
                   latitudes, longitudes)
        .get()
        .to(device=local_rank)
    )

    if cfg.world_size > 1:
        graphcast_model = DistributedDataParallel(
            graphcast_model, device_ids=[
                local_rank], output_device=local_rank
        )

    channels_list = [i for i in range(len(cfg.channels))]

    area = torch.abs(
        torch.cos(deg2rad(lat_lon_grid[:, :, 0])))
    area /= torch.mean(area)
    area = area.to(dtype=torch.bfloat16 if cfg.full_bf16 else torch.float32).to(
        device=local_rank
    )

    criterion = GraphCastLossFunction(
        area, channels_list, cfg.dataset_metadata_path, cfg.time_diff_std_path
    )
    optimizer = optimizers.FusedAdam(
        graphcast_model.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.95),
        adam_w_mode=True,
        weight_decay=0.1,
    )
    scheduler1 = LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=cfg.num_iters_step1,
    )
    scheduler2 = CosineAnnealingLR(
        optimizer, T_max=cfg.num_iters_step2, eta_min=0.0)
    scheduler3 = LambdaLR(
        optimizer, lr_lambda=lambda epoch: (cfg.lr_step3 / cfg.lr))
    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler1, scheduler2, scheduler3],
        milestones=[cfg.num_iters_step1,
                    cfg.num_iters_step1 + cfg.num_iters_step2],
    )

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    train_loss_file = f"{cfg.checkpoint_dir}/trloss.npy"
    world_rank == 0 and logger.info(f"start training ...")

    best_valid_loss = 1.0e6
    best_loss_epoch = 0
    train_losses = np.empty((0,), dtype=np.float32)

    # also can set it to 'len(train_dataloader) // 64'
    print_length = 1
    epoch_start_time = time.perf_counter()

    for epoch in range(cfg.num_iters_step1 + cfg.num_iters_step2):
        if dist.is_initialized():
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
        train_loss = 0.0
        for i, data in enumerate(train_dataloader):
            batch_start_time = time.perf_counter()
            graphcast_model.train()
            invar = data[0].to(device=local_rank)
            outvar = data[1].to(device=local_rank)
            cos_zenith = data[2].to(device=local_rank)
            in_idx = data[3].item()

            cos_zenith = torch.squeeze(cos_zenith, dim=2)
            cos_zenith = torch.clamp(
                cos_zenith, min=0.0) - 1.0 / torch.pi
            day_of_year, time_of_day = divmod(
                in_idx * cfg.dt, 24)
            normalized_day_of_year = torch.tensor(
                (day_of_year / 365) * (np.pi / 2),
                dtype=torch.float32,
                device=local_rank,
            )
            normalized_time_of_day = torch.tensor(
                (time_of_day / (24 - cfg.dt)) * (np.pi / 2),
                dtype=torch.float32,
                device=local_rank,
            )
            sin_day_of_year = torch.sin(
                normalized_day_of_year).expand(1, 1, 721, 1440)
            cos_day_of_year = torch.cos(
                normalized_day_of_year).expand(1, 1, 721, 1440)
            sin_time_of_day = torch.sin(
                normalized_time_of_day).expand(1, 1, 721, 1440)
            cos_time_of_day = torch.cos(
                normalized_time_of_day).expand(1, 1, 721, 1440)
            invar = torch.concat(
                (
                    invar,
                    cos_zenith,
                    static_data,
                    sin_day_of_year,
                    cos_day_of_year,
                    sin_time_of_day,
                    cos_time_of_day,
                ),
                dim=1,
            )

            invar, outvar = invar.to(
                dtype=model_dtype), outvar.to(dtype=model_dtype)
            outvar_pred = graphcast_model(invar)
            loss = criterion(outvar_pred, outvar)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                graphcast_model.parameters(), cfg.grad_clip_norm
            )
            torch.cuda.nvtx.range_pop()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

            if world_rank == 0 and i % print_length == 0:
                batch_time = time.perf_counter() - batch_start_time
                logger.info(
                    f"Epoch [{epoch + 1}/{cfg.max_epoch}], Train MiniBatch {i}/{len(train_dataloader)} done, "
                    f"This MiniBatch Cost: {batch_time / print_length:.2f}s, Current Loss: {loss.item():.4f}"
                )

            if (i + 1) % cfg.val_freq == 0:
                graphcast_model.eval()
                valid_loss = 0.0
                with torch.no_grad():
                    val_batch_time = time.perf_counter()
                    for j, data in enumerate(val_dataloader):
                        invar = data[0].to(
                            device=local_rank)
                        outvar = data[1].to(
                            device=local_rank)
                        cos_zenith = data[2].to(
                            device=local_rank)
                        in_idx = data[3].item()

                        cos_zenith = torch.squeeze(
                            cos_zenith, dim=2)
                        cos_zenith = (
                            torch.clamp(
                                cos_zenith, min=0.0) - 1.0 / torch.pi
                        )  # [b, 2, h, w]
                        outvar = outvar.to(
                            dtype=model_dtype)
                        loss = 0.0
                        for t in range(outvar.shape[1]):
                            day_of_year, time_of_day = divmod(
                                in_idx + t * cfg.dt, 24 // cfg.dt
                            )
                            normalized_day_of_year = torch.tensor(
                                (day_of_year / 365) *
                                (np.pi / 2),
                                dtype=torch.float32,
                                device=local_rank,
                            )
                            normalized_time_of_day = torch.tensor(
                                (time_of_day / (24 - cfg.dt)
                                 ) * (np.pi / 2),
                                dtype=torch.float32,
                                device=local_rank,
                            )
                            sin_day_of_year = torch.sin(normalized_day_of_year).expand(
                                1, 1, 721, 1440
                            )
                            cos_day_of_year = torch.cos(normalized_day_of_year).expand(
                                1, 1, 721, 1440
                            )
                            sin_time_of_day = torch.sin(normalized_time_of_day).expand(
                                1, 1, 721, 1440
                            )
                            cos_time_of_day = torch.cos(normalized_time_of_day).expand(
                                1, 1, 721, 1440
                            )
                            invar = torch.concat(
                                (
                                    invar,
                                    cos_zenith[:,
                                               t: t + 1, :, :],
                                    static_data,
                                    sin_day_of_year,
                                    cos_day_of_year,
                                    sin_time_of_day,
                                    cos_time_of_day,
                                ),
                                dim=1,
                            )
                            invar = invar.to(
                                dtype=model_dtype)
                            outpred = graphcast_model(invar)
                            invar = outpred
                            loss += criterion(outpred,
                                              outvar[:, t])

                        loss /= outvar.shape[1]
                        if cfg.world_size > 1:
                            loss_tensor = loss.detach().to(
                                local_rank
                            )  # torch.tensor(loss, device=local_rank)
                            dist.all_reduce(loss_tensor)
                            loss = loss_tensor.item() / cfg.world_size
                            valid_loss += loss
                        else:
                            valid_loss += loss.item()

                        if world_rank == 0 and j % print_length == 0:
                            val_batch_time = time.perf_counter() - val_batch_time
                            logger.info(
                                f"Epoch [{epoch + 1}/{cfg.max_epoch}], Val MiniBatch {j}/{len(val_dataloader)} done, "
                                f"This {cfg.num_val_steps}-step iter val process cost: {val_batch_time / print_length:.2f}s, "
                                f"Current Loss: {loss:.4f}"
                            )
                            val_batch_time = time.perf_counter()

                    valid_loss /= len(val_dataloader)
                    is_save_ckp = False
                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        best_loss_epoch = i
                        world_rank == 0 and save_checkpoint(
                            graphcast_model,
                            optimizer,
                            scheduler,
                            best_valid_loss,
                            best_loss_epoch,
                            cfg.checkpoint_dir,
                        )
                        is_save_ckp = True
                        if world_rank == 0:
                            logger.info(
                                f"Best loss at Minibatch: {i + 1}"
                                + (", saving checkpoint" if is_save_ckp else "")
                            )

        epoch_time = time.perf_counter() - epoch_start_time
        if world_rank == 0:
            logger.info(
                f"Epoch [{epoch + 1}/{cfg.max_epoch}] finished in {epoch_time:.2f}s, "
                f"Train Loss: {train_loss:.4f}, "
                f"This Epoch cost {time.perf_counter() - epoch_start_time: .2f}s, "
                f"Best loss at Minibatch: {i + 1}"
                + (", saving checkpoint" if is_save_ckp else "")
            )
            train_losses = np.append(
                train_losses, train_loss)

            np.save(train_loss_file, train_losses)
        if epoch - best_loss_epoch > cfg.patience:
            print(
                f"Loss has not decrease in {cfg.patience} epochs, stopping training..."
            )
            exit()

    print(
        f"Graphcast has been well-trained, next step is fine-tune")


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
    torch.save(state, f"{model_path}/graphcast.pth")


if __name__ == "__main__":
    current_path = os.getcwd()
    sys.path.append(current_path)
    main()
