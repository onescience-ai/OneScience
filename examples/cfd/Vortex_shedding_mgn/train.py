# train_meshgraphnet_dgl.py
# 
# 适配 DGL 的 OneScience 训练脚本
# [已修正] 使用 LambdaLR 调度器

import os
import sys
import logging
import time
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.amp import GradScaler, autocast

from onescience.distributed.manager import DistributedManager

# --- 本地 imports ---
from onescience.utils.YParams import YParams
from onescience.datapipes import DeepMind_CylinderFlowDatapipe
from onescience.launch.utils import load_checkpoint, save_checkpoint 
from onescience.models.meshgraphnet import MeshGraphNet


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


def main():
    DistributedManager.initialize()
    manager = DistributedManager()
    logger = setup_logging(manager.rank)
    
    # 1. 加载配置
    config_file_path = "conf/mgn_cylinderflow.yaml"
    cfg = YParams(config_file_path, "model")
    cfg_data = YParams(config_file_path, "datapipe")
    cfg_train = YParams(config_file_path, "training")
    
    log_interval = getattr(cfg_train, "log_interval", 100)
    
    # --- 动态模型选择 ---
    model_name = cfg.name
    if manager.rank == 0:
        logger.info(f"===== 🚀 Preparing model: {model_name} (DGL) =====")
        logger.info(f"Training logs will be printed every {log_interval} batches.")

    if model_name not in cfg.specific_params:
        raise ValueError(f"Model '{model_name}' not found in config's 'specific_params' block.")
    model_params = cfg.specific_params[model_name]

    # 2. 初始化 Datapipe (DGL 版本)
    logger.info("Initializing datapipe (DGL version)...")
    datapipe = DeepMind_CylinderFlowDatapipe(params=cfg_data, distributed=(manager.world_size > 1))
    train_dataloader, train_sampler = datapipe.train_dataloader()
    val_dataloader, val_sampler = datapipe.val_dataloader()
    
    stats = datapipe.stats
    logger.info("Datapipe initialized.")

    # 3. 设置 Device (与 PyG 版本相同)
    if manager.world_size > 1:
        device = torch.device(f'cuda:{manager.local_rank}' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(f'cuda:{cfg_train.gpuid}' if torch.cuda.is_available() else 'cpu')
        
    # 4. 初始化模型 (与 PyG 版本相同, 使用修正后的关键字)
    logger.info(f"Initializing model architecture: {model_name}")
    
    if model_name == 'MeshGraphNet':
        mlp_act = "relu"
        if model_params.recompute_activation:
            if manager.rank == 0:
                logger.info("Setting MLP activation to SiLU for recompute_activation.")
            mlp_act = "silu"

        model = MeshGraphNet(
            input_dim_nodes=model_params.num_input_features,
            input_dim_edges=model_params.num_edge_features,
            output_dim=model_params.num_output_features,
            mlp_activation_fn=mlp_act,
            do_concat_trick=model_params.do_concat_trick,
            num_processor_checkpoint_segments=model_params.num_processor_checkpoint_segments,
            recompute_activation=model_params.recompute_activation,
        ).to(device)

    else:
        raise NotImplementedError(f"Model {model_name} initialization not implemented.")

    if manager.rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model: {model_name}, Trainable Params: {total_params / 1e6:.2f}M")
        
    if cfg_train.jit:
        if hasattr(model, 'meta') and not model.meta.jit: 
            logger.warning("MeshGraphNet JIT support not explicitly enabled.")
        try:
            model = torch.jit.script(model).to(device)
            logger.info("Model JIT compilation successful.")
        except Exception as e:
            logger.error(f"Model JIT compilation failed: {e}")
            logger.warning("Falling back to non-JIT model.")
            model = model.to(device)

    if manager.world_size > 1:
        model = DistributedDataParallel(
            model, 
            device_ids=[manager.local_rank], 
            output_device=manager.local_rank,
            find_unused_parameters=True 
        )

    # 5. 初始化优化器、调度器、损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg_train.lr)
    

    if not hasattr(cfg_train, 'lr_decay_rate'):
         raise ValueError("Missing 'lr_decay_rate' in training config for LambdaLR.")
         
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda=lambda step: cfg_train.lr_decay_rate**step 
    )
    # --- [修正结束] ---
    
    if cfg_train.loss_criterion == 'MSE':
        loss_criterion = nn.MSELoss()
    elif cfg_train.loss_criterion == 'MAE':
        loss_criterion = nn.L1Loss()
    else:
        raise ValueError(f"Unknown loss_criterion: {cfg_train.loss_criterion}")

    scaler = GradScaler() if cfg_train.amp else None

    # 6. 加载 Checkpoint
    checkpoint_dir = cfg_train.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    epoch_init = 0
    if manager.world_size > 1:
        torch.distributed.barrier()
        
    epoch_init = load_checkpoint(
        checkpoint_dir,
        models=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
    )
    if epoch_init > 0:
         logger.info(f"Loaded checkpoint. Resuming training from epoch {epoch_init}")


    # 7. 训练循环
    best_valid_loss = 1.0e6
    best_loss_epoch = 0

    logger.info("Starting training...")
    for epoch in range(epoch_init, cfg_train.max_epoch):
        epoch_start_time = time.time()
        
        if manager.world_size > 1:
            train_sampler.set_epoch(epoch)
            if val_sampler: val_sampler.set_epoch(epoch)
            
        model.train()
        train_loss_sum = 0.0 
        
        for idx, data in enumerate(train_dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            
            with autocast(device_type=device.type, enabled=cfg_train.amp):
                out = model(data.ndata["x"], data.edata["x"], data)
                targets = data.ndata["y"]
                loss = loss_criterion(out, targets)
            
            if cfg_train.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                
            # LambdaLR (像 OneCycleLR 一样) 应该在每个 "批次" 更新
            # 这与您原始 MGNTrainer 的逻辑一致
            scheduler.step()
            
            current_batch_loss = loss.item()
            train_loss_sum += current_batch_loss

            if manager.rank == 0 and (idx + 1) % log_interval == 0:
                avg_loss_so_far = train_loss_sum / (idx + 1)
                total_batches = len(train_dataloader)
                
                logger.info(
                    f"  Epoch [{epoch + 1}/{cfg_train.max_epoch}] | "
                    f"Batch [{idx + 1}/{total_batches}] | "
                    f"Batch Loss: {current_batch_loss:.6f} | "
                    f"Epoch Avg Loss: {avg_loss_so_far:.6f}"
                )
            
        train_loss_avg = train_loss_sum / len(train_dataloader)
        
        # --- 验证 ---
        model.eval()
        valid_loss = 0
        
        with torch.no_grad():
            for data in val_dataloader:
                data = data.to(device)
                
                with autocast(device_type=device.type, enabled=cfg_train.amp):
                    out = model(data.ndata["x"], data.edata["x"], data)
                    targets = data.ndata["y"]
                    loss = loss_criterion(out, targets)
                
                if manager.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.AVG)
                    
                valid_loss += loss.item()
                
        valid_loss /= len(val_dataloader)

        # --- 日志和 Checkpointing ---
        if manager.rank == 0:
            epoch_time = time.time() - epoch_start_time
            logger.info(
                f"Epoch [{epoch + 1}/{cfg_train.max_epoch}] | Time: {epoch_time:.2f}s | "
                f"Train Loss: {train_loss_avg:.6f} | "
                f"Valid Loss: {valid_loss:.6f}"
            )
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_loss_epoch = epoch
                
                save_checkpoint(
                    checkpoint_dir,
                    models=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch + 1,
                )
                logger.info(f"  -> New best validation loss. Checkpoint saved.")

            if (epoch - best_loss_epoch) > cfg_train.patience:
                logger.warning(f"Validation loss has not improved for {cfg_train.patience} epochs. Stopping training.")
                break
                
    # 8. 训练后测试
    if manager.rank == 0:
        logger.info("===== ✅ Training finished. =====")
        # ...

if __name__ == "__main__":
    main()