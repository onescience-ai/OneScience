# train_transolver_car.py

import os
import sys
import logging
import time
import json
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from onescience.distributed.manager import DistributedManager

from onescience.utils.YParams import YParams
from onescience.datapipes import ShapeNetCarDatapipe
from onescience.models.transolver.Transolver3D import Transolver3D


def setup_logging(rank):
    """设置日志，只在 rank 0 输出 INFO"""
    level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=level, 
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logging.getLogger().setLevel(level)
    return logging.getLogger()

def save_checkpoint(model, optimizer, scheduler, epoch, loss, ckp_dir, model_name):
    """保存 checkpoint"""
    if not os.path.exists(ckp_dir):
        os.makedirs(ckp_dir, exist_ok=True)
        
    model_to_save = model.module if hasattr(model, "module") else model
    state = {
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }
    torch.save(state, f"{ckp_dir}/{model_name}.pth")

def main():
    DistributedManager.initialize()
    manager = DistributedManager()
    logger = setup_logging(manager.rank)
    
    # 加载配置
    config_file_path = "conf/transolver_car.yaml"
    cfg = YParams(config_file_path, "model")
    cfg_data = YParams(config_file_path, "datapipe")
    cfg_train = YParams(config_file_path, "training")
    
    # --- 动态模型选择 ---
    model_name = cfg.name
    if manager.rank == 0:
        logger.info(f"===== Preparing model: {model_name} =====")

    if model_name not in cfg.specific_params:
        raise ValueError(f"Model '{model_name}' not found in config's 'specific_params' block.")
    model_params = cfg.specific_params[model_name]

    # 将模型特定的数据参数 (hparams) 注入 datapipe 配置
    cfg_data.model_hparams = model_params
    # -------------------------
    
    # 初始化 Datapipe
    logger.info("Initializing datapipe...")
    datapipe = ShapeNetCarDatapipe(params=cfg_data, distributed=(manager.world_size > 1)) 
    train_dataloader, train_sampler = datapipe.train_dataloader()
    val_dataloader, val_sampler = datapipe.val_dataloader()
    
    # 获取 coef_norm
    coef_norm = datapipe.coef_norm
    logger.info("Datapipe initialized.")

    # 设置 Device
    if manager.world_size > 1:
        device = torch.device(f'cuda:{manager.local_rank}' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(f'cuda:{cfg_train.gpuid}' if torch.cuda.is_available() else 'cpu')
        
    # 初始化模型
    logger.info(f"Initializing model architecture: {model_name}")
    
    if model_name == 'Transolver':
        # Transolver 3D
        model = Transolver3D(
            n_hidden=model_params.n_hidden,
            n_layers=model_params.n_layers,
            space_dim=model_params.space_dim,
            fun_dim=model_params.fun_dim,
            n_head=model_params.n_head,
            mlp_ratio=model_params.mlp_ratio,
            out_dim=model_params.out_dim,
            slice_num=model_params.slice_num,
            unified_pos=model_params.unified_pos
        ).to(device)
    else:
        raise NotImplementedError(f"Model {model_name} initialization not implemented.")

    if manager.rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model: {model_name}, Trainable Params: {total_params / 1e6:.2f}M")
        
    if manager.world_size > 1:
        model = DistributedDataParallel(
            model, 
            device_ids=[manager.local_rank], 
            output_device=manager.local_rank,
            find_unused_parameters=True 
        )

    # 初始化优化器、调度器、损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg_train.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg_train.lr,
        total_steps=(len(train_dataloader)) * cfg_train.max_epoch,
    )
    
    if cfg_train.loss_criterion == 'MSE':
        loss_criterion = nn.MSELoss(reduction='none')
    elif cfg_train.loss_criterion == 'MAE':
        loss_criterion = nn.L1Loss(reduction='none')
    else:
        raise ValueError(f"Unknown loss criterion: {cfg_train.loss_criterion}")
        
    loss_weight = cfg_train.loss_weight # (reg)
    
    # 训练循环
    checkpoint_dir = cfg_train.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_valid_loss = 1.0e6
    best_loss_epoch = 0
    hparams_dict = cfg_train.to_dict() # 用于保存 log.json

    logger.info("Starting training...")
    for epoch in range(cfg_train.max_epoch):
        epoch_start_time = time.time()
        if manager.world_size > 1:
            train_sampler.set_epoch(epoch)
            if val_sampler: val_sampler.set_epoch(epoch)
            
        model.train()
        train_loss = 0
        train_loss_press = 0
        train_loss_velo = 0
        
        # --- 训练循环---
        for data in train_dataloader:
            data = data.to(device)
            optimizer.zero_grad()
            
            out = model(data) 
            #out = model((data, data))
            targets = data.y
            
            # --- ShapeNetCar 特定损失函数 ---
            loss_press_vec = loss_criterion(out[data.surf, -1], targets[data.surf, -1])
            loss_press = loss_press_vec.mean()
            
            loss_velo_vec = loss_criterion(out[:, :-1], targets[:, :-1])
            loss_velo = loss_velo_vec.mean()

            loss = loss_velo + loss_weight * loss_press
            # ---------------------------------
                
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            train_loss_press += loss_press.item()
            train_loss_velo += loss_velo.item()
            
        train_loss /= len(train_dataloader)
        train_loss_press /= len(train_dataloader)
        train_loss_velo /= len(train_dataloader)
        
        # --- 验证 ---
        model.eval()
        valid_loss = 0
        valid_loss_press = 0
        valid_loss_velo = 0
        
        # --- 验证循环  ---
        if (epoch + 1) % cfg_train.val_iter == 0 or epoch == cfg_train.max_epoch - 1:
            with torch.no_grad():
                for data in val_dataloader:
                    data = data.to(device)
                    out = model(data)
                    targets = data.y

                    # --- ShapeNetCar 特定损失函数 ---
                    loss_press = loss_criterion(out[data.surf, -1], targets[data.surf, -1]).mean()
                    loss_velo = loss_criterion(out[:, :-1], targets[:, :-1]).mean()
                    loss = loss_velo + loss_weight * loss_press
                    # ---------------------------------
                    
                    if manager.world_size > 1:
                        dist.all_reduce(loss, op=dist.ReduceOp.AVG)
                        dist.all_reduce(loss_press, op=dist.ReduceOp.AVG)
                        dist.all_reduce(loss_velo, op=dist.ReduceOp.AVG)
                        
                    valid_loss += loss.item()
                    valid_loss_press += loss_press.item()
                    valid_loss_velo += loss_velo.item()
                    
            valid_loss /= len(val_dataloader)
            valid_loss_press /= len(val_dataloader)
            valid_loss_velo /= len(val_dataloader)

        # --- 日志和 Checkpointing ---
        if manager.rank == 0:
            epoch_time = time.time() - epoch_start_time
            log_msg = (
                f"Epoch [{epoch + 1}/{cfg_train.max_epoch}] | Time: {epoch_time:.2f}s | "
                f"Train Loss: {train_loss:.6f} (Velo: {train_loss_velo:.6f}, Press: {train_loss_press:.6f})"
            )

            if valid_loss > 0:
                 log_msg += f" | Valid Loss: {valid_loss:.6f} (Velo: {valid_loss_velo:.6f}, Press: {valid_loss_press:.6f})"

            logger.info(log_msg)
            
            is_save_ckp = False
            if valid_loss > 0 and valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_loss_epoch = epoch
                save_checkpoint(model, optimizer, scheduler, epoch, valid_loss, checkpoint_dir, model_name)
                is_save_ckp = True
                logger.info(f"   -> New best validation loss. Checkpoint saved.")

            if epoch - best_loss_epoch > cfg_train.patience:
                logger.warning(f"Validation loss has not improved for {cfg_train.patience} epochs. Stopping training.")
                break
                

if __name__ == "__main__":
    main()