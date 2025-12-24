# %% [markdown]
# # Model
# ## 1. 主要功能
# 进行模块化拆分，在数据处理、模型结构、损失函数以及日志记录等方面形成独立模块，各子模块通过标准化接口进行解耦与交互，实现训练流程的低耦合，为后续模型调优与组件复用奠定基础；<br>
# 
# **模型结构**<br>
# 常用的基于Transformer架构的深度学习预测模型，通过自注意力机制实现多尺度气象特征的高效建模，用于对天气变化进行精准预测；

# %%
## 模型基础库

import torch
import os
import sys
import numpy as np
import logging
import time
import math
import torch.distributed as dist
from dataclasses import dataclass
from torch.nn.parallel import DistributedDataParallel

# %%
## 模型结构

from onescience.memory.checkpoint import replace_function
from onescience.metrics import L1_loss
from onescience.optimizers import FusedAdam
from onescience.models.meta import ModelMetaData
from onescience.models.module import Module

# %%
## 日志信息模块
def main():
    
    current_path = os.path.dirname(os.path.abspath(__file__))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger()

    # %%
    ## 配置文件模块

    from onescience.utils.YParams import YParams

    config_file_path = os.path.join(current_path, "conf/config.yaml")
    cfg = YParams(config_file_path, "model")
    cfg.world_size = 1
    if "WORLD_SIZE" in os.environ:
        cfg.world_size = int(os.environ["WORLD_SIZE"])

    world_rank = 0
    local_rank = 0

    if cfg.world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ["LOCAL_RANK"])
        world_rank = dist.get_rank()

    # %%
    ## 数据模块

    from onescience.datapipes.climate import ERA5HDF5Datapipe

    land_mask = torch.from_numpy(np.load(os.path.join(cfg.static_dir, "land_mask.npy")).astype(np.float32))
    soil_type = torch.from_numpy(np.load(os.path.join(cfg.static_dir, "soil_type.npy")).astype(np.float32))
    topography = torch.from_numpy(np.load(os.path.join(cfg.static_dir, "topography.npy")).astype(np.float32))
    surface_mask = torch.stack([land_mask, soil_type, topography], dim=0).to(local_rank)
    surface_mask = surface_mask.unsqueeze(0).repeat(cfg.batch_size, 1, 1, 1)

    train_dataset = ERA5HDF5Datapipe(params=cfg, distributed=dist.is_initialized())
    train_dataloader, train_sampler = train_dataset.train_dataloader()

    val_dataset = ERA5HDF5Datapipe(params=cfg, distributed=dist.is_initialized())
    val_dataloader, val_sampler = val_dataset.val_dataloader()

    # %%
    ## 数据信息

    @dataclass
    class MetaData(ModelMetaData):
        name: str = "Pangu"
        # Optimization
        jit: bool = False  # ONNX Ops Conflict
        cuda_graphs: bool = True
        amp: bool = True
        # Inference
        onnx_cpu: bool = False  # No FFT op on CPU
        onnx_gpu: bool = True
        onnx_runtime: bool = True
        # Physics informed
        var_dim: int = 1
        func_torch: bool = False
        auto_grad: bool = False

    # %%
    ## 模型模块

    from onescience.modules import (
        PatchRecovery2D,
        PatchRecovery3D,
        FuserLayer,
        DownSample3D,
        UpSample3D,
        PatchEmbed2D,
        PatchEmbed3D,
    )

    class Pangu(Module):
        """
        Pangu A PyTorch impl of: `Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Global Weather Forecast`
        - https://arxiv.org/abs/2211.02556

        Args:
            img_size (tuple[int]): Image size [Lat, Lon].
            patch_size (tuple[int]): Patch token size [Lat, Lon].
            embed_dim (int): Patch embedding dimension. Default: 192
            num_heads (tuple[int]): Number of attention heads in different layers.
            window_size (tuple[int]): Window size.
        """

        def __init__(
            self,
            img_size=(721, 1440),
            patch_size=(2, 4, 4),
            embed_dim=192,
            num_heads=(6, 12, 12, 6),
            window_size=(2, 6, 12),
        ):
            super().__init__(meta=MetaData())
            drop_path = np.linspace(0, 0.2, 8).tolist()
            # In addition, three constant masks(the topography mask, land-sea mask and soil type mask)
            self.patchembed2d = PatchEmbed2D(
                img_size=img_size,
                patch_size=patch_size[1:],
                in_chans=4 + 3,  # add
                embed_dim=embed_dim,
            )
            self.patchembed3d = PatchEmbed3D(
                img_size=(13, img_size[0], img_size[1]),
                patch_size=patch_size,
                in_chans=5,
                embed_dim=embed_dim,
            )
            patched_inp_shape = (
                8,
                math.ceil(img_size[0] / patch_size[1]),
                math.ceil(img_size[1] / patch_size[2]),
            )

            self.layer1 = FuserLayer(
                dim=embed_dim,
                input_resolution=patched_inp_shape,
                depth=2,
                num_heads=num_heads[0],
                window_size=window_size,
                drop_path=drop_path[:2],
            )

            patched_inp_shape_downsample = (
                8,
                math.ceil(patched_inp_shape[1] / 2),
                math.ceil(patched_inp_shape[2] / 2),
            )
            self.downsample = DownSample3D(
                in_dim=embed_dim,
                input_resolution=patched_inp_shape,
                output_resolution=patched_inp_shape_downsample,
            )
            self.layer2 = FuserLayer(
                dim=embed_dim * 2,
                input_resolution=patched_inp_shape_downsample,
                depth=6,
                num_heads=num_heads[1],
                window_size=window_size,
                drop_path=drop_path[2:],
            )
            self.layer3 = FuserLayer(
                dim=embed_dim * 2,
                input_resolution=patched_inp_shape_downsample,
                depth=6,
                num_heads=num_heads[2],
                window_size=window_size,
                drop_path=drop_path[2:],
            )
            self.upsample = UpSample3D(
                embed_dim * 2, embed_dim, patched_inp_shape_downsample, patched_inp_shape
            )
            self.layer4 = FuserLayer(
                dim=embed_dim,
                input_resolution=patched_inp_shape,
                depth=2,
                num_heads=num_heads[3],
                window_size=window_size,
                drop_path=drop_path[:2],
            )
            # The outputs of the 2nd encoder layer and the 7th decoder layer are concatenated along the channel dimension.
            self.patchrecovery2d = PatchRecovery2D(
                img_size, patch_size[1:], 2 * embed_dim, 4
            )
            self.patchrecovery3d = PatchRecovery3D(
                (13, img_size[0], img_size[1]), patch_size, 2 * embed_dim, 5
            )

        def prepare_input(self, surface, surface_mask, upper_air):
            """Prepares the input to the model in the required shape.
            Args:
                surface (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=4.
                surface_mask (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=3.
                upper_air (torch.Tensor): 3D n_pl=13, n_lat=721, n_lon=1440, chans=5.
            """
            upper_air = upper_air.reshape(
                upper_air.shape[0], -1, upper_air.shape[3], upper_air.shape[4]
            )
            surface_mask = surface_mask.unsqueeze(0).repeat(surface.shape[0], 1, 1, 1)
            return torch.concat([surface, surface_mask, upper_air], dim=1)

        def forward(self, x):
            """
            Args:
                x (torch.Tensor): [batch, 4+3+5*13, lat, lon]
            """
            surface = x[:, :7, :, :]
            upper_air = x[:, 7:, :, :].reshape(x.shape[0], 5, 13, x.shape[2], x.shape[3])
            surface = self.patchembed2d(surface)
            upper_air = self.patchembed3d(upper_air)

            x = torch.concat([surface.unsqueeze(2), upper_air], dim=2)
            B, C, Pl, Lat, Lon = x.shape
            x = x.reshape(B, C, -1).transpose(1, 2)
            x = self.layer1(x)
            skip = x
            x = self.downsample(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.upsample(x)
            x = self.layer4(x)

            output = torch.concat([x, skip], dim=-1)
            output = output.transpose(1, 2).reshape(B, -1, Pl, Lat, Lon)
            output_surface = output[:, :, 0, :, :]
            output_upper_air = output[:, :, 1:, :, :]

            output_surface = self.patchrecovery2d(output_surface)
            output_upper_air = self.patchrecovery3d(output_upper_air)
            return output_surface, output_upper_air

    pangu_model = Pangu(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            window_size=cfg.window_size,
        ).to(local_rank)

    if cfg.world_size > 1:
            pangu_model = DistributedDataParallel(pangu_model, device_ids=[local_rank], output_device=local_rank)

    # %%
    optimizer = FusedAdam(pangu_model.parameters(), betas=(0.9, 0.999), lr=5e-4, weight_decay=3e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # %%
    world_rank == 0 and logger.info(f"start training ...")
    best_valid_loss = 1.0e6
    best_loss_epoch = 0
    train_losses = np.empty((0,), dtype=np.float32)
    valid_losses = np.empty((0,), dtype=np.float32)

    # %%

    def save_checkpoint(
        model, optimizer, scheduler, best_valid_loss, best_loss_epoch, model_path
    ):
        model_to_save = model.module if hasattr(model, "module") else model
        state = {
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_valid_loss": best_valid_loss,
            "best_loss_epoch": best_loss_epoch,
        }
        torch.save(state, f"{model_path}/pangu_weather.pth")

    # %%
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    train_loss_file = f"{cfg.checkpoint_dir}/trloss.npy"
    valid_loss_file = f"{cfg.checkpoint_dir}/valoss.npy"

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
                invar_surface = invar[:, :4, :, :].to(local_rank, dtype=torch.float32)
                invar_upper_air = invar[:, 4:, :, :].to(local_rank, dtype=torch.float32)
                invar = torch.concat([invar_surface, surface_mask, invar_upper_air], dim=1)
                tar_surface = outvar[:, :4, :, :].to(local_rank, dtype=torch.float32)
                tar_upper_air = outvar[:, 4:, :, :].to(local_rank, dtype=torch.float32)

                with replace_function(
                    pangu_model,
                    ["layer1", "layer2", "layer3", "layer4"],
                    cfg.world_size > 1,
                ):
                    out_surface, out_upper_air = pangu_model(invar)

                out_upper_air = out_upper_air.reshape(tar_upper_air.shape)
                loss1 = L1_loss(tar_surface, out_surface)
                loss2 = L1_loss(tar_upper_air, out_upper_air)
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
                    invar_surface = invar[:, :4, :, :].to(local_rank, dtype=torch.float32)
                    invar_upper_air = invar[:, 4:, :, :].to(local_rank, dtype=torch.float32)
                    invar = torch.concat(
                        [invar_surface, surface_mask, invar_upper_air], dim=1
                    )

                    tar_surface = outvar[:, :4, :, :].to(local_rank, dtype=torch.float32)
                    tar_upper_air = outvar[:, 4:, :, :].to(local_rank, dtype=torch.float32)

                    out_surface, out_upper_air = pangu_model(invar)
                    out_upper_air = out_upper_air.reshape(tar_upper_air.shape)

                    loss1 = L1_loss(tar_surface, out_surface).item()
                    loss2 = L1_loss(tar_upper_air, out_upper_air).item()
                    loss = loss1 * 0.25 + loss2

                    if cfg.world_size > 1:
                        loss_tensor = torch.tensor(loss, device=local_rank)
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
                train_losses = np.append(train_losses, train_loss)
                valid_losses = np.append(valid_losses, valid_loss)

                np.save(train_loss_file, train_losses)
                np.save(valid_loss_file, valid_losses)
            if epoch - best_loss_epoch > cfg.patience:
                print(
                    f"Loss has not decrease in {cfg.patience} epochs, stopping training..."
                )
                exit()

if __name__ == "__main__":
    current_path = os.getcwd()
    sys.path.append(current_path)
    main()

# %%
