import os
import torch
import torch.nn as nn
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy

# Onescience imports
from onescience.utils.YParams import YParams
from onescience.distributed.manager import DistributedManager
from onescience.datapipes.cfd import DeepCFDDatapipe
from torch.nn.parallel import DistributedDataParallel as DDP

# 动态导入模型
def init_model(cfg):
    model_name = cfg.model.name
    if model_name == "UNet":
        from onescience.models.deepcfd.UNet import UNet
        net_class = UNet
    elif model_name == "UNetEx":
        from onescience.models.deepcfd.UNetEx import UNetEx
        net_class = UNetEx
    elif model_name == "AutoEncoder":
        from onescience.models.deepcfd.AutoEncoder import AutoEncoder
        net_class = AutoEncoder
    else:
        raise ValueError(f"Unknown network: {model_name}")
    
    model = net_class(
        cfg.model.in_channels,
        cfg.model.out_channels,
        filters=cfg.model.filters,
        kernel_size=cfg.model.kernel_size,
        batch_norm=cfg.model.batch_norm,
        weight_norm=cfg.model.weight_norm
    )
    return model

def loss_func(output, target, weights):
    """
    DeepCFD 自定义 Loss: Weighted MSE + Abs Error
    weights shape: (1, 3, 1, 1)
    """
    # output/target shape: (B, 3, H, W)
    # Channel 0: Ux, Channel 1: Uy, Channel 2: p
    
    # Ux MSE
    lossu = ((output[:, 0, :, :] - target[:, 0, :, :]) ** 2)
    # Uy MSE
    lossv = ((output[:, 1, :, :] - target[:, 1, :, :]) ** 2)
    # p Abs Error (原始代码逻辑如此)
    lossp = torch.abs((output[:, 2, :, :] - target[:, 2, :, :]))
    
    # Stack back to (B, 3, H, W) to apply weights
    loss_stack = torch.stack([lossu, lossv, lossp], dim=1)
    
    # Apply weights
    weighted_loss = loss_stack / weights
    
    return torch.sum(weighted_loss)

def evaluate(model, loader, device, weights, dist):
    model.eval()
    total_loss = 0.0
    total_ux_mse = 0.0
    total_uy_mse = 0.0
    total_p_mse = 0.0
    num_batches = 0
    
    with torch.no_grad():
        # 仅 rank 0 显示进度条
        iterator = tqdm(loader, desc="Evaluating", disable=(dist.rank != 0))
        for batch in iterator:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            
            output = model(x)
            
            # Loss
            loss = loss_func(output, y, weights)
            total_loss += loss.item()
            
            # Metrics (MSE for each channel)
            total_ux_mse += torch.sum((output[:, 0] - y[:, 0]) ** 2).item()
            total_uy_mse += torch.sum((output[:, 1] - y[:, 1]) ** 2).item()
            total_p_mse += torch.sum((output[:, 2] - y[:, 2]) ** 2).item()
            
            num_batches += 1
            
    avg_loss = total_loss / num_batches 
    return avg_loss, total_ux_mse, total_uy_mse, total_p_mse

def main():
    # 1. Initialize
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device
    
    # 2. Config
    config_path = "conf/deepcfd.yaml"
    cfg = YParams(config_path, "root")
    
    output_dir = Path(cfg.training.output_dir)
    if dist.rank == 0:
        print(f"Loading config from {config_path}")
        print(f"Output directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        # 保存参数
        # cfg.save(str(output_dir / "config.yaml"))

    # 3. Data
    if dist.rank == 0: print("Initializing Datapipe...")
    datapipe = DeepCFDDatapipe(cfg.datapipe, distributed=(dist.world_size > 1))
    train_loader, train_sampler = datapipe.train_dataloader()
    test_loader, test_sampler = datapipe.test_dataloader()
    
    # 获取 loss 权重
    loss_weights = datapipe.get_loss_weights().to(device)
    if dist.rank == 0:
        print(f"Loss weights: {loss_weights.view(-1).cpu().numpy()}")

    # 4. Model
    if dist.rank == 0: print(f"Initializing Model: {cfg.model.name}")
    model = init_model(cfg).to(device)
    
    if dist.world_size > 1:
        model = DDP(model, device_ids=[dist.local_rank], output_device=dist.local_rank)

    # 5. Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg.training.lr, 
        weight_decay=cfg.training.weight_decay
    )

    # 6. Training Loop
    if dist.rank == 0: print("Starting Training...")
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(cfg.training.num_epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
            
        model.train()
        train_loss = 0.0
        
        # 仅 rank 0 显示进度条
        iterator = tqdm(train_loader, desc=f"Epoch {epoch}", disable=(dist.rank != 0))
        
        for batch in iterator:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = loss_func(output, y, loss_weights)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Update tqdm bar
            if dist.rank == 0:
                iterator.set_postfix({"loss": f"{loss.item():.4e}"})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Evaluation & Saving
        if (epoch + 1) % cfg.training.eval_interval == 0:
            val_loss, ux_err, uy_err, p_err = evaluate(model, test_loader, device, loss_weights, dist)
            
            if dist.rank == 0:
                print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4e} | Val Loss: {val_loss:.4e}")
                print(f"Metrics (Sum Sq Err): Ux={ux_err:.2e}, Uy={uy_err:.2e}, P={p_err:.2e}")
                
                # Early Stopping Logic & Saving Best Model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # Save Best
                    model_to_save = model.module if hasattr(model, "module") else model
                    ckpt = {
                        "model_state": model_to_save.state_dict(),
                        "config": cfg.model.to_dict(), 
                        "epoch": epoch
                    }
                    torch.save(ckpt, output_dir / "best_model.pt")
                    print("--> Saved Best Model")
                else:
                    patience_counter += 1
            
            # Sync stop flag
            stop_flag = torch.tensor([0], device=device)
            if dist.rank == 0 and patience_counter >= cfg.training.patience:
                print("Early stopping triggered.")
                stop_flag += 1
            
            if dist.world_size > 1:
                torch.distributed.broadcast(stop_flag, src=0)
                
            if stop_flag.item() > 0:
                break

    dist.cleanup()

if __name__ == "__main__":
    main()