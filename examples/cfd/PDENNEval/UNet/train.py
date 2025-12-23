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
from onescience.datapipes import PDEBenchUNetDatapipe
from torch.nn.parallel import DistributedDataParallel as DDP
from onescience.models.pdenneval.unet import UNet1d, UNet2d, UNet3d

def get_model(spatial_dim, cfg):
    assert spatial_dim <= 3, "Spatial dimension of data can not exceed 3."
    model_args = cfg.model
    initial_step = cfg.datapipe.data.initial_step
    
    if spatial_dim == 1:
        model = UNet1d(model_args.in_channels * initial_step, 
                       model_args.out_channels, 
                       model_args.init_features)
    elif spatial_dim == 2:
        model = UNet2d(model_args.in_channels * initial_step, 
                       model_args.out_channels, 
                       model_args.init_features)
    elif spatial_dim == 3:
        model = UNet3d(model_args.in_channels * initial_step, 
                       model_args.out_channels, 
                       model_args.init_features)
    return model

def train_loop(dataloader, model, loss_fn, optimizer, device, cfg, rank, pbar=None):
    model.train()
    step_losses = []

    train_cfg = cfg.training
    data_cfg = cfg.datapipe.data

    initial_step = data_cfg.initial_step
    unroll_step = train_cfg.unroll_step

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        batch_size = x.size(0)
        t_train = y.shape[-2]

        loss = 0
        pred = y[..., :initial_step, :]
        input_shape = list(x.shape)[:-2]
        input_shape.append(-1)

        for t in range(initial_step, t_train):
            if train_cfg.training_type != "autoregressive":
                model_in_tensor = y[..., t-initial_step:t, :]
            else:
                model_in_tensor = x

            model_input = model_in_tensor.reshape(input_shape)

            input_permute = [0, -1] + [i for i in range(1, len(model_input.shape)-1)]
            model_input = model_input.permute(input_permute)

            output_permute = [0] + [i for i in range(2, len(model_input.shape))] + [1]

            target = y[..., t:t+1, :]

            if t < t_train - unroll_step:
                with torch.no_grad():
                    model_output = model(model_input).permute(output_permute).unsqueeze(-2)
            else:
                model_output = model(model_input).permute(output_permute).unsqueeze(-2)
                _loss = loss_fn(model_output.reshape(batch_size, -1), target.reshape(batch_size, -1))
                loss += _loss

            pred = torch.cat((pred, model_output), dim=-2)

            if train_cfg.training_type == "autoregressive":
                x = torch.cat((x[..., 1:, :], model_output), dim=-2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val = float(loss.item())
        step_losses.append(loss_val)

        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss_val:.3e}")

    return float(np.mean(step_losses)) if len(step_losses) else 0.0


@torch.no_grad()
def val_loop(dataloader, model, loss_fn, device, cfg):
    model.eval()
    full_losses = []
    
    train_cfg = cfg.training
    data_cfg = cfg.datapipe.data
    initial_step = data_cfg.initial_step
    
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        batch_size = x.size(0)
        t_train = y.shape[-2]
        
        pred = y[..., :initial_step, :]
        input_shape = list(x.shape)[:-2]
        input_shape.append(-1)
        
        for t in range(initial_step, t_train):
            model_input = x.reshape(input_shape)
            
            # Permute
            input_permute = [0, -1] + [i for i in range(1, len(model_input.shape)-1)]
            model_input = model_input.permute(input_permute)
            
            output_permute = [0] + [i for i in range(2, len(model_input.shape))] + [1]
            
            model_output = model(model_input).permute(output_permute).unsqueeze(-2)
            
            pred = torch.cat((pred, model_output), dim=-2)
            x = torch.cat((x[..., 1:, :], model_output), dim=-2)
            
        loss = loss_fn(pred.reshape(batch_size, -1), y.reshape(batch_size, -1))
        full_losses.append(loss.item())
        
    return np.mean(full_losses)

def main():
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device

    parser = argparse.ArgumentParser(description='Train UNet model')
    parser.add_argument('config', type=str, help='Path to config file')
    args = parser.parse_args()
    config_path = args.config
    # 验证配置文件是否存在
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
        
    cfg = YParams(config_path, "unet_config")
    
    output_dir = Path(cfg.training.output_dir)
    if dist.rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output: {output_dir}")

    # Datapipe
    datapipe = PDEBenchUNetDatapipe(cfg, distributed=(dist.world_size > 1))
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
        if train_sampler:
            train_sampler.set_epoch(epoch)
    
        pbar = None
        if dist.rank == 0:
            total = len(train_loader) if hasattr(train_loader, "__len__") else None
            pbar = tqdm(
                total=total,
                desc=f"Epoch {epoch+1}/{cfg.training.epochs}",
                dynamic_ncols=True,
                leave=False,
                mininterval=0.5,
            )
    
        train_loss = train_loop(train_loader, model, loss_fn, optimizer, device, cfg, dist.rank, pbar=pbar)
    
        if pbar is not None:
            pbar.close()
    
        scheduler.step()
    
        if dist.rank == 0:
            tqdm.write(f"[Epoch {epoch}] Train Loss: {train_loss:.4e}")
    
            if (epoch + 1) % cfg.training.save_period == 0:
                val_loss = val_loop(val_loader, model, loss_fn, device, cfg)
                tqdm.write(f"[Epoch {epoch}] Val Loss: {val_loss:.4e}")
    
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
                    tqdm.write(f"Saved Best Model to {best_path}")

    dist.cleanup()

if __name__ == "__main__":
    main()