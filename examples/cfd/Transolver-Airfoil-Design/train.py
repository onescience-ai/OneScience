# train_transolver.py

import os
import sys
import logging
import time
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from onescience.distributed.manager import DistributedManager

# 本地 imports
from onescience.utils.YParams import YParams
from onescience.datapipes import AirfRANSDatapipe
import onescience.utils.transolver.metrics as metrics # 导入 metrics

# --- 导入所有需要的模型 ---
from onescience.models.transolver.Transolver2D import Transolver2D
from onescience.models.transolver.MLP import MLP
from onescience.models.transolver.GraphSAGE import GraphSAGE # 假设路径
from onescience.models.transolver.PointNet import PointNet   # 假设路径
from onescience.models.transolver.NN import NN           # 假设路径
from onescience.models.transolver.GUNet import GUNet       # 假设路径
# -------------------------

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
    # 使用模型名称保存，避免冲突
    torch.save(state, f"{ckp_dir}/{model_name}.pth")

def main():
    DistributedManager.initialize()
    manager = DistributedManager()
    logger = setup_logging(manager.rank)
    
    # 1. 加载配置
    config_file_path = "conf/transolver_airfrans.yaml"
    cfg = YParams(config_file_path, "model")
    cfg_data = YParams(config_file_path, "datapipe")
    cfg_train = YParams(config_file_path, "training")
    
    # --- 动态模型选择 ---
    model_name = cfg.name
    if manager.rank == 0:
        logger.info(f"===== 🚀 Preparing model: {model_name} =====")

    # 获取该模型的所有特定参数
    if model_name not in cfg.specific_params:
        raise ValueError(f"Model '{model_name}' not found in config's 'specific_params' block.")
    model_params = cfg.specific_params[model_name]


    # 将模型特定的数据参数 (hparams) 注入 datapipe 配置
    cfg_data.model_hparams = model_params
    # -------------------------
    hparams = model_params
    if not hasattr(hparams, 'subsampling') or hparams.subsampling is None:
            hparams.subsampling = cfg_data.data.subsampling
            logger.info(f"Added 'subsampling = {hparams.subsampling}' to hparams for Infer_test.")

    # 2. 初始化 Datapipe
    logger.info("Initializing datapipe...")
    datapipe = AirfRANSDatapipe(params=cfg_data, distributed=(manager.world_size > 1))
    train_dataloader, train_sampler = datapipe.train_dataloader()
    val_dataloader, val_sampler = datapipe.val_dataloader()
    
    # 获取 coef_norm 以用于后续测试
    coef_norm = datapipe.coef_norm
    logger.info("Datapipe initialized.")

    # 3. 设置 Device
    if manager.world_size > 1:
        device = torch.device(f'cuda:{manager.local_rank}' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(f'cuda:{cfg_train.gpuid}' if torch.cuda.is_available() else 'cpu')
        
    # 4. 初始化模型 (动态)
    logger.info(f"Initializing model architecture: {model_name}")
    
    if model_name == 'Transolver':
        # Transolver 有不同的初始化签名
        model = Transolver2D(
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
        # 其他模型共享 encoder/decoder 结构
        # 注意: YParams 节点需要转换为 python 列表
        encoder = MLP(list(model_params.encoder), batch_norm=False)
        decoder = MLP(list(model_params.decoder), batch_norm=False)

        if model_name == 'GraphSAGE':
            # 将 YParams 节点转换为 dict
            model = GraphSAGE(model_params.to_dict(), encoder, decoder).to(device)
        elif model_name == 'PointNet':
            model = PointNet(model_params.to_dict(), encoder, decoder).to(device)
        elif model_name == 'MLP':
            model = NN(model_params.to_dict(), encoder, decoder).to(device)
        elif model_name == 'GUNet':
            model = GUNet(model_params.to_dict(), encoder, decoder).to(device)
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
            find_unused_parameters=True # PyG 模型通常需要
        )

    # 5. 初始化优化器、调度器、损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg_train.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg_train.lr,
        total_steps=(len(train_dataloader)) * cfg_train.max_epoch,
    )
    
    if cfg_train.loss_criterion == 'MSE' or cfg_train.loss_criterion == 'MSE_weighted':
        loss_criterion = nn.MSELoss(reduction='none')
    elif cfg_train.loss_criterion == 'MAE':
        loss_criterion = nn.L1Loss(reduction='none')
    loss_weight = cfg_train.loss_weight
    use_weighted_loss = (cfg_train.loss_criterion == 'MSE_weighted')
    
    # 6. 训练循环
    checkpoint_dir = cfg_train.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_valid_loss = 1.0e6
    best_loss_epoch = 0

    logger.info("Starting training...")
    for epoch in range(cfg_train.max_epoch):
        epoch_start_time = time.time()
        if manager.world_size > 1:
            train_sampler.set_epoch(epoch)
            if val_sampler: val_sampler.set_epoch(epoch)
            
        model.train()
        train_loss = 0
        train_loss_surf = 0
        train_loss_vol = 0
        
        for data in train_dataloader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            targets = data.y
            
            loss_all_nodes = loss_criterion(out, targets).mean(dim=0)
            loss_surf_var = loss_criterion(out[data.surf], targets[data.surf]).mean(dim=0)
            loss_vol_var = loss_criterion(out[~data.surf], targets[~data.surf]).mean(dim=0)
            loss_surf = loss_surf_var.mean()
            loss_vol = loss_vol_var.mean()

            if use_weighted_loss:
                loss = loss_vol + loss_weight * loss_surf
            else:
                loss = loss_all_nodes.mean()
                
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            train_loss_surf += loss_surf.item()
            train_loss_vol += loss_vol.item()
            
        train_loss /= len(train_dataloader)
        train_loss_surf /= len(train_dataloader)
        train_loss_vol /= len(train_dataloader)
        
        # --- 验证 ---
        model.eval()
        valid_loss = 0
        valid_loss_surf = 0
        valid_loss_vol = 0
        
        with torch.no_grad():
            for data in val_dataloader:
                data = data.to(device)
                out = model(data)
                targets = data.y

                loss_surf = loss_criterion(out[data.surf], targets[data.surf]).mean()
                loss_vol = loss_criterion(out[~data.surf], targets[~data.surf]).mean()
                
                if use_weighted_loss:
                    loss = loss_vol + loss_weight * loss_surf
                else:
                    loss = loss_criterion(out, targets).mean()
                
                if manager.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.AVG)
                    dist.all_reduce(loss_surf, op=dist.ReduceOp.AVG)
                    dist.all_reduce(loss_vol, op=dist.ReduceOp.AVG)
                    
                valid_loss += loss.item()
                valid_loss_surf += loss_surf.item()
                valid_loss_vol += loss_vol.item()
                
        valid_loss /= len(val_dataloader)
        valid_loss_surf /= len(val_dataloader)
        valid_loss_vol /= len(val_dataloader)

        # --- 日志和 Checkpointing (仅 Rank 0) ---
        if manager.rank == 0:
            epoch_time = time.time() - epoch_start_time
            logger.info(
                f"Epoch [{epoch + 1}/{cfg_train.max_epoch}] | Time: {epoch_time:.2f}s | "
                f"Train Loss: {train_loss:.6f} (Vol: {train_loss_vol:.6f}, Surf: {train_loss_surf:.6f}) | "
                f"Valid Loss: {valid_loss:.6f} (Vol: {valid_loss_vol:.6f}, Surf: {valid_loss_surf:.6f})"
            )
            
            is_save_ckp = False
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_loss_epoch = epoch
                save_checkpoint(model, optimizer, scheduler, epoch, valid_loss, checkpoint_dir, model_name)
                is_save_ckp = True
                logger.info(f"   -> New best validation loss. Checkpoint saved.")

            if epoch - best_loss_epoch > cfg_train.patience:
                logger.warning(f"Validation loss has not improved for {cfg_train.patience} epochs. Stopping training.")
                break
                
    # 7. 训练后测试 (仅在 Rank 0 上执行)
    if manager.rank == 0:
        logger.info("===== ✅ Training finished. Starting testing... =====")
        
        # 加载最佳模型
        best_model_path = f"{checkpoint_dir}/{model_name}.pth"
        if os.path.exists(best_model_path):
            logger.info(f"Loading best checkpoint from: {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location=device)
            # 确保 DDP 和非 DDP 加载都兼容
            model_to_test = model.module if hasattr(model, "module") else model
            model_to_test.load_state_dict(checkpoint['model_state_dict'])
            models = [model_to_test]
        else:
            logger.warning("No checkpoint found. Testing with the final model state.")
            models = [model.module if hasattr(model, "module") else model]

        # 将 hparams (YParams 节点) 转换为 dict 
        hparams_for_metrics = model_params.to_dict()
        
        results_dir = checkpoint_dir
        
        # 调用 metrics 函数
        coefs = metrics.Results_test(
            device,
            models,
            [hparams_for_metrics], # 原始代码期望一个列表
            coef_norm,
            cfg_data.source.data_dir,
            results_dir,
            cfg_train.n_test,
            criterion=cfg_train.loss_criterion,
            s=cfg_data.data.splits.test_name
        )
        logger.info(f"Testing complete. Results saved in: {results_dir}")
        
        # (可选) 保存结果
        np.save(os.path.join(results_dir, 'true_coefs'), coefs[0])
        np.save(os.path.join(results_dir, 'pred_coefs_mean'), coefs[1])
        # ... (保存 coefs[2] 到 coefs[6])

if __name__ == "__main__":
    main()