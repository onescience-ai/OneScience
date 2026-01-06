import os
import sys
import torch
import torch.nn as nn
import numpy as np
import copy
from timeit import default_timer
from pathlib import Path
from tqdm import tqdm
import argparse
from onescience.utils.YParams import YParams
from onescience.distributed.manager import DistributedManager
from onescience.datapipes.cfd.PDENNEval import PDEBenchUNODatapipe
from torch.nn.parallel import DistributedDataParallel as DDP

# UNO Models
from onescience.models.pdenneval.uno import UNO1d, UNO2d, UNO3d, UNO_maxwell

def get_model(spatial_dim, cfg):
    assert spatial_dim <= 3, "Spatial dimension of data can not exceed 3."
    model_args = cfg.model
    initial_step = cfg.datapipe.data.initial_step
    
    if cfg.training.pde_name == "3D_Maxwell":
        model = UNO_maxwell(num_channels=model_args.num_channels,
                            width=model_args.width,
                            initial_step=initial_step)
    elif spatial_dim == 1:
        model = UNO1d(num_channels=model_args.num_channels,
                      width=model_args.width,
                      initial_step=initial_step)
    elif spatial_dim == 2:
        model = UNO2d(num_channels=model_args.num_channels,
                      width=model_args.width,
                      initial_step=initial_step)
    elif spatial_dim == 3:
        model = UNO3d(num_channels=model_args.num_channels,
                      width=model_args.width,
                      initial_step=initial_step)
    return model

def train_loop(dataloader, model, loss_fn, optimizer, device, cfg, rank, pbar=None):
    model.train()
    train_l2 = 0.0
    train_l_inf = 0.0

    train_cfg = cfg.training
    data_cfg  = cfg.datapipe.data

    initial_step = data_cfg.initial_step
    t_train      = train_cfg.t_train

    for x, y, grid in dataloader:
        # 0) 只取 cfg 需要的时间长度，避免 y 过长/不一致导致 reshape 尺寸变化
        y = y[..., :t_train, :]

        # Maxwell 预处理
        if train_cfg.pde_name == "3D_Maxwell":
            grid_scale = grid[0]
            x = x.detach().clone() / grid_scale
            y = y.detach().clone() / grid_scale

        x = x.to(device)       # (bs, x1..., init_t, v)
        y = y.to(device)       # (bs, x1..., t_train, v)
        grid = grid.to(device)

        batch_size = x.size(0)
        loss = 0.0

        pred = y[..., :initial_step, :]   # (bs, x1..., init_t, v)

        input_shape = list(x.shape)[:-2]
        input_shape.append(-1)            # (bs, x1..., init_t*v)

        if train_cfg.training_type == "autoregressive":
            for t in range(initial_step, t_train):
                model_input = x.reshape(input_shape)         # (bs, x1..., init_t*v)
                target = y[..., t:t+1, :]                    # (bs, x1..., 1, v)

                model_output = model(model_input, grid)      # 可能是 (bs, x1..., v) 或 (bs, x1..., 1, v)
                if model_output.dim() == target.dim() - 1:   # 补齐时间维
                    model_output = model_output.unsqueeze(-2)

                # 1) loss 必须是 单步输出 vs 单步 target
                loss = loss + loss_fn(
                    model_output.reshape(batch_size, -1),
                    target.reshape(batch_size, -1)
                )

                pred = torch.cat((pred, model_output), dim=-2)
                x = torch.cat((x[..., 1:, :], model_output), dim=-2)

            train_l2 += float(loss.item())
            train_l_inf = max(
                train_l_inf,
                torch.max(torch.abs(pred.reshape(batch_size, -1) - y.reshape(batch_size, -1))).item()
            )

        elif train_cfg.training_type == "single":
            # 单步训练：只对最后一步做监督
            model_input_single = x[..., 0, :]                # (bs, x1..., v)
            target = y[..., t_train-1:t_train, :]            # (bs, x1..., 1, v)

            pred_single = model(model_input_single, grid)    # 可能是 (bs, x1..., v) 或 (bs, x1..., 1, v)
            if pred_single.dim() == target.dim() - 1:
                pred_single = pred_single.unsqueeze(-2)

            loss = loss_fn(
                pred_single.reshape(batch_size, -1),
                target.reshape(batch_size, -1)               # 2) 这里一定用 target，不要用 y
            )

            train_l2 += float(loss.item())
            train_l_inf = max(
                train_l_inf,
                torch.max(torch.abs(pred_single.reshape(batch_size, -1) - target.reshape(batch_size, -1))).item()
            )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(loss=f"{float(loss.item()):.3e}")

    train_l2 /= max(len(dataloader), 1)
    return train_l2, train_l_inf


@torch.no_grad()
def val_loop(dataloader, model, loss_fn, device, cfg):
    model.eval()
    val_l2_full = 0.0
    val_l_inf_full = 0.0

    train_cfg = cfg.training
    data_cfg  = cfg.datapipe.data

    initial_step = data_cfg.initial_step
    t_train_cfg  = train_cfg.t_train

    for x, y, grid in dataloader:
        # 关键：验证也对齐 t_train，避免 y 太长/不一致
        t_total = min(t_train_cfg, y.shape[-2])
        if initial_step >= t_total:
            continue
        y = y[..., :t_total, :]

        # Maxwell 预处理
        if train_cfg.pde_name == "3D_Maxwell":
            grid_scale = grid[0]
            x = x.detach().clone() / grid_scale
            y = y.detach().clone() / grid_scale

        x = x.to(device)
        y = y.to(device)
        grid = grid.to(device)
        batch_size = x.size(0)

        input_shape = list(x.shape)[:-2]
        input_shape.append(-1)

        if train_cfg.training_type == 'autoregressive':
            pred = y[..., :initial_step, :]

            for t in range(initial_step, t_total):
                model_input = x.reshape(input_shape)
                model_output = model(model_input, grid)

                # 关键：补齐 time 维，保证 cat 的 dim=-2 是时间维
                if model_output.dim() == pred.dim() - 1:
                    model_output = model_output.unsqueeze(-2)

                pred = torch.cat((pred, model_output), dim=-2)
                x = torch.cat((x[..., 1:, :], model_output), dim=-2)

            # 只在对齐的区间算 loss / Linf
            _pred = pred[..., initial_step:t_total, :]
            _y    = y[..., initial_step:t_total, :]

            val_l2_full += loss_fn(_pred.reshape(batch_size, -1), _y.reshape(batch_size, -1)).item()
            val_l_inf_full = max(
                val_l_inf_full,
                torch.max(torch.abs(_pred.reshape(batch_size, -1) - _y.reshape(batch_size, -1))).item()
            )

        elif train_cfg.training_type == 'single':
            model_input_single = x[..., 0, :]
            target = y[..., t_total-1:t_total, :]

            pred = model(model_input_single, grid)
            if pred.dim() == target.dim() - 1:
                pred = pred.unsqueeze(-2)

            val_l2_full += loss_fn(pred.reshape(batch_size, -1), target.reshape(batch_size, -1)).item()
            val_l_inf_full = max(
                val_l_inf_full,
                torch.max(torch.abs(pred.reshape(batch_size, -1) - target.reshape(batch_size, -1))).item()
            )

    val_l2_full /= max(len(dataloader), 1)
    return val_l2_full, val_l_inf_full


def main():
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device
    
    parser = argparse.ArgumentParser(description='Train UNO model')
    parser.add_argument('config', type=str, help='Path to config file')
    args = parser.parse_args()
    config_path = args.config
    # 验证配置文件是否存在
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
        
    cfg = YParams(config_path, "uno_config")
    
    output_dir = Path(cfg.training.output_dir)
    if dist.rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output: {output_dir}")

    # Datapipe
    datapipe = PDEBenchUNODatapipe(cfg, distributed=(dist.world_size > 1))
    train_loader, train_sampler = datapipe.train_dataloader()
    val_loader, val_sampler = datapipe.val_dataloader()
    
    spatial_dim = datapipe.spatial_dim
    
    # Model
    model = get_model(spatial_dim, cfg).to(device)
    if dist.world_size > 1:
        model = DDP(model, device_ids=[dist.local_rank], output_device=dist.local_rank)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.optimizer.lr, weight_decay=cfg.training.optimizer.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.training.scheduler.step_size, gamma=cfg.training.scheduler.gamma)
    loss_fn = nn.MSELoss()
    
    if dist.rank == 0: print("Starting Training...")
    
    min_val_loss = float('inf')
    
    for epoch in range(cfg.training.epochs):
        if train_sampler: train_sampler.set_epoch(epoch)
        
        train_loss, train_l_inf = train_loop(train_loader, model, loss_fn, optimizer, device, cfg, dist.rank)
        scheduler.step()
        
        if dist.rank == 0:
            print(f"[Epoch {epoch}] Train Loss: {train_loss:.4e} Linf: {train_l_inf:.4e}")
            
            if (epoch + 1) % cfg.training.save_period == 0:
                val_loss, val_l_inf = val_loop(val_loader, model, loss_fn, device, cfg)
                print(f"[Epoch {epoch}] Val Loss: {val_loss:.4e} Val Linf: {val_l_inf:.4e}")
                
                ckpt_path = output_dir / f"model_epoch_{epoch}.pt"
                model_state = model.module.state_dict() if dist.world_size > 1 else model.state_dict()
                
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": val_loss
                }, ckpt_path)
                
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    best_path = output_dir / "best_model.pt"
                    torch.save(model_state, best_path)
                    print(f"Saved Best Model to {best_path}")

    dist.cleanup()

if __name__ == "__main__":
    main()