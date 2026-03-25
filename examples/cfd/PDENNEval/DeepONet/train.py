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
from onescience.datapipes.cfd.PDENNEval import PDEBenchDeepONetDatapipe
from torch.nn.parallel import DistributedDataParallel as DDP

# 引用现有 DeepONet 模型
from onescience.models.pdenneval.deeponet import DeepONetCartesianProd2D, DeepONetCartesianProd1D
from onescience.utils.pdenneval.deeponet_utils import count_params, to_device

def get_model(spatial_dim, if_temporal, cfg):
    assert spatial_dim <= 3, "Spatial dimension of data can not exceed 3."
    model_args = cfg.model
    initial_step = cfg.datapipe.data.initial_step
    
    if spatial_dim == 1:
        model = DeepONetCartesianProd1D(
            size=model_args.input_size,
            in_channel_branch=model_args.in_channels * initial_step,
            query_dim=model_args.query_dim,
            out_channel=model_args.out_channels,
            activation=model_args.act,
            base_model=model_args.base_model
        )
    elif spatial_dim == 2:
        model = DeepONetCartesianProd2D(
            size=model_args.input_size,
            in_channel_branch=model_args.in_channels * initial_step,
            query_dim=model_args.query_dim,
            out_channel=model_args.out_channels,
            activation=model_args.act,
            base_model=model_args.base_model
        )
    else:
        raise NotImplementedError("3D not supported yet.")
        
    return model

def train_loop(train_loader, model, optimizer, device, cfg, rank):
    model.train()
    train_loss = 0.0
    train_l_inf = 0.0
    loss_fn = nn.MSELoss(reduction="mean")
    
    train_cfg = cfg.training
    data_cfg = cfg.datapipe.data
    
    initial_step = data_cfg.initial_step
    if_temporal = cfg.runtime.if_temporal 
    
    for x, y, grid in train_loader:
        if torch.any(y.isnan()): continue
        
        bs = y.shape[0]
        t_train = y.shape[-2]
        grid = grid[0] # Reduce batch for Grid
        x, y, grid = to_device([x, y, grid], device)
        
        input_shape = list(x.shape)[:-2]
        input_shape.append(-1)
        
        loss = 0
        
        if if_temporal:
            if train_cfg.training_type == 'single':
                pred = model((x.reshape(input_shape), grid))
                loss += loss_fn(pred.reshape(bs, -1), y.reshape(bs, -1))
                train_loss += loss.item()
            else: # autoregressive
                grid_in = grid[..., 0, :-1] # slice grid for time? (Original logic)
                pred = y[..., :initial_step, :]
                
                for t in range(initial_step, t_train):
                    model_input = x.reshape(input_shape)
                    target = y[..., t:t+1, :]
                    
                    # Model run
                    model_output = model((model_input, grid_in)).unsqueeze(-2)
                    
                    _loss = loss_fn(model_output.reshape(bs, -1), target.reshape(bs, -1))
                    loss += _loss
                    
                    pred = torch.cat((pred, model_output), -2)
                    x = torch.cat((x[..., 1:, :], model_output), dim=-2)
                
                train_loss += loss_fn(pred.reshape(bs, -1), y.reshape(bs, -1)).item()
                train_l_inf = torch.max(torch.abs(pred.reshape(bs, -1) - y.reshape(bs, -1))).item()
                
        else: # DarcyFlow (Steady state)
            a = y[..., 0, 0:1] # Original logic: specific slicing for Darcy
            u = y[..., 0, 1:2]
            grid_in = grid[..., 0, :-1]
            pred = model((a, grid_in))
            
            loss += loss_fn(pred.reshape(bs, -1), u.reshape(bs, -1))
            train_loss += loss.item()
            train_l_inf = max(train_l_inf, torch.max(torch.abs(pred.reshape(bs, -1) - u.reshape(bs, -1))).item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    train_loss /= len(train_loader)
    return train_loss, train_l_inf

@torch.no_grad()
def val_loop(val_loader, model, device, cfg):
    model.eval()
    val_loss = 0.0
    val_l_inf = 0.0
    loss_fn = nn.MSELoss(reduction="mean")
    
    train_cfg = cfg.training
    data_cfg = cfg.datapipe.data
    initial_step = data_cfg.initial_step
    if_temporal = cfg.runtime.if_temporal
    
    for x, y, grid in val_loader:
        if torch.any(y.isnan()): continue
        
        bs = y.shape[0]
        t_train = y.shape[-2]
        grid = grid[0]
        x, y, grid = to_device([x, y, grid], device)
        
        pred = y[..., :initial_step, :]
        input_shape = list(x.shape)[:-2]
        input_shape.append(-1)
        
        if if_temporal:
            if train_cfg.training_type == 'single':
                pred = model((x.reshape(input_shape), grid))
                val_loss += loss_fn(pred.reshape(bs, -1), y.reshape(bs, -1)).item()
            else:
                grid_in = grid[..., 0, :-1]
                for t in range(initial_step, t_train):
                    model_input = x.reshape(input_shape)
                    model_output = model((model_input, grid_in)).unsqueeze(-2)
                    pred = torch.cat((pred, model_output), -2)
                    x = torch.cat((x[..., 1:, :], model_output), dim=-2)
                
                val_loss += loss_fn(pred.reshape(bs, -1), y.reshape(bs, -1)).item()
                val_l_inf = max(val_l_inf, torch.max(torch.abs(pred.reshape(bs, -1) - y.reshape(bs, -1))).item())
        else:
            a = y[..., 0, 0:1]
            u = y[..., 0, 1:2]
            grid_in = grid[..., 0, :-1]
            pred = model((a, grid_in))
            val_loss += loss_fn(pred.reshape(bs, -1), u.reshape(bs, -1)).item()
            val_l_inf = max(val_l_inf, torch.max(torch.abs(pred.reshape(bs, -1) - u.reshape(bs, -1))).item())
            
    val_loss /= len(val_loader)
    return val_loss, val_l_inf

def main():
    # 1. 初始化分布式环境
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device

    # 2. 解析命令行参数 
    parser = argparse.ArgumentParser(description='Train DeepONet model')
    parser.add_argument('config', type=str, help='Path to config file')
    args = parser.parse_args()
    config_path = args.config

    # 验证配置文件是否存在
    if not os.path.exists(config_path):
        if dist.rank == 0:
            print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
        
    cfg = YParams(config_path, "deeponet_config")

    if hasattr(cfg.datapipe, 'source') and hasattr(cfg.datapipe.source, 'data_dir'):
        cfg.datapipe.source.data_dir = os.path.expandvars(cfg.datapipe.source.data_dir)
    # ---------------------------------------

    output_dir = Path(cfg.training.output_dir)
    if dist.rank == 0:
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output Directory: {output_dir}")
        print(f"Training on {dist.world_size} GPUs")
        print(f"Data Directory after expansion: {cfg.datapipe.source.data_dir}")

    # 4. Data Pipeline
    datapipe = PDEBenchDeepONetDatapipe(cfg, distributed=(dist.world_size > 1))
    train_loader, train_sampler = datapipe.train_dataloader()
    val_loader, val_sampler = datapipe.val_dataloader()
    
    # 动态获取数据维度逻辑
    temp_sample = next(iter(train_loader))
    _, sample_y, _ = temp_sample
    spatial_dim = len(sample_y.shape) - 3
    if_temporal = True if sample_y.shape[-2] != 1 else False
    
    # 将运行时推断的状态存入 cfg
    cfg.runtime = type('obj', (object,), {'if_temporal': if_temporal})
    
    # 5. Model
    model = get_model(spatial_dim, if_temporal, cfg).to(device)
    if dist.rank == 0:
        print(f"Model Params: {count_params(model)}")
        
    if dist.world_size > 1:
        model = DDP(model, device_ids=[dist.local_rank], output_device=dist.local_rank)
        
    # 6. Optimizer & Scheduler
    train_cfg = cfg.training
    optim_cfg = train_cfg.optimizer
    sched_cfg = train_cfg.scheduler
    
    optimizer = getattr(torch.optim, optim_cfg.name)(
        model.parameters(), 
        lr=optim_cfg.lr, 
        weight_decay=optim_cfg.weight_decay
    )
    scheduler = getattr(torch.optim.lr_scheduler, sched_cfg.name)(
        optimizer, 
        step_size=sched_cfg.step_size, 
        gamma=sched_cfg.gamma
    )
    
    # 7. Resume Logic
    start_epoch = 0
    min_val_loss = float('inf')
    if train_cfg.continue_training:
        ckpt = torch.load(train_cfg.model_path, map_location=device)
        model_to_load = model.module if dist.world_size > 1 else model
        model_to_load.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt['epoch']
        min_val_loss = ckpt['loss']
        if dist.rank == 0: print(f"Resumed from epoch {start_epoch}")

    # 8. Training Loop
    if dist.rank == 0:
        print("Starting Training...")
        pbar = tqdm(range(start_epoch, train_cfg.epochs), dynamic_ncols=True)
    else:
        pbar = range(start_epoch, train_cfg.epochs)

    for epoch in pbar:
        if train_sampler:
            train_sampler.set_epoch(epoch)
            
        train_loss, train_l_inf = train_loop(train_loader, model, optimizer, device, cfg, dist.rank)
        scheduler.step()
        
        if dist.rank == 0:
            pbar.set_description(f"[Epoch {epoch}] Loss: {train_loss:.4e} Linf: {train_l_inf:.4e}")
            
            # Save Latest
            ckpt_path = output_dir / f"{train_cfg.save_name}_latest.pt"
            model_state = model.module.state_dict() if dist.world_size > 1 else model.state_dict()
            torch.save({
                "epoch": epoch+1,
                "loss": min_val_loss,
                "model_state_dict": model_state,
                "optimizer_state_dict": optimizer.state_dict()
            }, ckpt_path)
            
            # Validation
            if (epoch + 1) % train_cfg.save_period == 0:
                val_loss, val_l_inf = val_loop(val_loader, model, device, cfg)
                print(f"\n[Val Epoch {epoch}] Val Loss: {val_loss:.4e} Val Linf: {val_l_inf:.4e}")
                
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    best_path = output_dir / f"{train_cfg.save_name}_best.pt"
                    torch.save({
                        "epoch": epoch+1,
                        "loss": min_val_loss,
                        "model_state_dict": model_state,
                        "optimizer_state_dict": optimizer.state_dict()
                    }, best_path)
                    print(f"Saved Best Model to {best_path}")

    if dist.rank == 0:
        print("Training Done.")
    dist.cleanup()

if __name__ == "__main__":
    main()