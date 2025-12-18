import os
import sys
import torch
import torch.nn as nn
import numpy as np
from timeit import default_timer
from pathlib import Path
import argparse
from onescience.utils.YParams import YParams
from onescience.distributed.manager import DistributedManager
from onescience.datapipes import PDEBenchFNODatapipe
from torch.nn.parallel import DistributedDataParallel as DDP

from onescience.utils.pdenneval.fno_utils import *
from onescience.models.pdenneval.fno import FNO1d, FNO2d, FNO3d, FNO_maxwell

def get_model(spatial_dim, cfg):
    assert spatial_dim <= 3, "Spatial dimension can not exceed 3."
    model_args = cfg.model
    # initial_step 现在在 datapipe.data 中
    initial_step = cfg.datapipe.data.initial_step
    
    if cfg.datapipe.data.pde_name == "3D_Maxwell":
        model = FNO_maxwell(
            num_channels=model_args.num_channels,
            width=model_args.width,
            modes1=model_args.modes,
            modes2=model_args.modes,
            modes3=model_args.modes,
            initial_step=initial_step
        )
    elif spatial_dim == 1:
        model = FNO1d(
            num_channels=model_args.num_channels,
            width=model_args.width,
            modes=model_args.modes,
            initial_step=initial_step
        )
    elif spatial_dim == 2:
        model = FNO2d(
            num_channels=model_args.num_channels,
            width=model_args.width,
            modes1=model_args.modes,
            modes2=model_args.modes,
            initial_step=initial_step
        )
    elif spatial_dim == 3:
        model = FNO3d(
            num_channels=model_args.num_channels,
            width=model_args.width,
            modes1=model_args.modes,
            modes2=model_args.modes,
            modes3=model_args.modes,
            initial_step=initial_step
        )
    return model

def train_loop(model, train_loader, optimizer, loss_fn, scheduler, device, cfg, rank):
    model.train()
    t1 = default_timer()
    train_l2 = 0.0
    
    # 常用参数提取
    data_cfg = cfg.datapipe.data
    train_cfg = cfg.training
    
    initial_step = data_cfg.initial_step
    t_train = train_cfg.t_train
    
    for x, y, grid in train_loader:
        x, y, grid = x.to(device), y.to(device), grid.to(device)
        
        if data_cfg.pde_name == "3D_Maxwell":
            grid_scale = grid[0]
            x = x / grid_scale
            y = y / grid_scale
        
        loss = 0
        pred = y[..., :initial_step, :]
        input_shape = list(x.shape)[:-2] 
        input_shape.append(-1) 

        if train_cfg.training_type == 'autoregressive':
            for t in range(initial_step, t_train):
                model_input = x.reshape(input_shape)
                target = y[..., t:t+1, :]
                model_output = model(model_input, grid)
                loss += loss_fn(model_output.reshape(model_output.size(0), -1), target.reshape(target.size(0), -1))
                pred = torch.cat((pred, model_output), -2)
                x = torch.cat((x[..., 1:, :], model_output), dim=-2)
        
        elif train_cfg.training_type == 'single':
            model_input = x.reshape(input_shape) 
            target = y[..., t_train-1:t_train, :]
            pred = model(model_input, grid)
            loss += loss_fn(pred.reshape(pred.size(0), -1), target.reshape(target.size(0), -1))
        
        train_l2 += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    scheduler.step()
    t2 = default_timer()
    return train_l2, t2 - t1

def val_loop(val_loader, model, loss_fn, device, cfg):
    model.eval()
    val_l2 = 0.0
    
    data_cfg = cfg.datapipe.data
    train_cfg = cfg.training
    initial_step = data_cfg.initial_step
    t_train = train_cfg.t_train

    with torch.no_grad():
        for x, y, grid in val_loader:
            x, y, grid = x.to(device), y.to(device), grid.to(device)
            
            if data_cfg.pde_name == "3D_Maxwell":
                grid_scale = grid[0]
                x = x / grid_scale
                y = y / grid_scale

            input_shape = list(x.shape)[:-2]
            input_shape.append(-1)

            if train_cfg.training_type == 'autoregressive':
                pred = y[..., :initial_step, :]
                for t in range(initial_step, y.shape[-2]): 
                    model_input = x.reshape(input_shape)
                    model_output = model(model_input, grid)
                    pred = torch.cat((pred, model_output), -2)
                    x = torch.cat((x[..., 1:, :], model_output), dim=-2)
                
                _pred = pred[..., initial_step:t_train, :]
                _y = y[..., initial_step:t_train, :]
                val_l2 += loss_fn(_pred.reshape(_y.size(0), -1), _y.reshape(_y.size(0), -1)).item()

            elif train_cfg.training_type == 'single':
                model_input = x.reshape(input_shape)
                target = y[..., t_train-1:t_train, :]
                pred = model(model_input, grid)
                val_l2 += loss_fn(pred.reshape(target.size(0), -1), target.reshape(target.size(0), -1)).item()

    return val_l2

def main():
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device

    parser = argparse.ArgumentParser(description='Train FNO model')
    parser.add_argument('config', type=str, help='Path to config file')
    args = parser.parse_args()
    config_path = args.config
    # 验证配置文件是否存在
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
        
    cfg = YParams(config_path, "fno_config")

    # [!!! 修复处 !!!] 从 training 节点下获取 output_dir
    output_dir = Path(cfg.training.output_dir)
    
    if dist.rank == 0:
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output Directory: {output_dir}")
        print(f"Training on {dist.world_size} GPUs")

    # 初始化 Datapipe
    datapipe = PDEBenchFNODatapipe(cfg, distributed=(dist.world_size > 1))
    train_loader, train_sampler = datapipe.train_dataloader()
    val_loader, val_sampler = datapipe.val_dataloader()

    spatial_dim = datapipe.spatial_dim
    model = get_model(spatial_dim, cfg).to(device)
    
    if dist.world_size > 1:
        model = DDP(model, device_ids=[dist.local_rank], output_device=dist.local_rank)

    # 提取训练相关配置
    train_cfg = cfg.training
    optim_cfg = train_cfg.optimizer
    sched_cfg = train_cfg.scheduler

    optimizer_cls = getattr(torch.optim, optim_cfg.name)
    optimizer = optimizer_cls(model.parameters(), lr=optim_cfg.lr, weight_decay=optim_cfg.weight_decay)
    
    scheduler_cls = getattr(torch.optim.lr_scheduler, sched_cfg.name)
    scheduler = scheduler_cls(optimizer, step_size=sched_cfg.step_size, gamma=sched_cfg.gamma)
    
    loss_fn = nn.MSELoss()

    if dist.rank == 0:
        print("Starting Training...")
    
    for epoch in range(train_cfg.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
            
        train_l2, time_taken = train_loop(
            model, train_loader, optimizer, loss_fn, scheduler, device, cfg, dist.rank
        )
        
        # Logging (Rank 0 only)
        if dist.rank == 0:
            print(f"[Epoch {epoch}] Train L2: {train_l2:.5f}, Time: {time_taken:.2f}s")
            
            if (epoch + 1) % train_cfg.save_period == 0:
                val_l2 = val_loop(val_loader, model, loss_fn, device, cfg)
                print(f"[Epoch {epoch}] Val L2: {val_l2:.5f}")
                
                ckpt_path = output_dir / f"model_epoch_{epoch}.pt"
                model_to_save = model.module if hasattr(model, "module") else model
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model_to_save.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": val_l2
                }, ckpt_path)
    
    if dist.rank == 0:
        print("Training Done.")
    
    dist.cleanup()

if __name__ == "__main__":
    main()