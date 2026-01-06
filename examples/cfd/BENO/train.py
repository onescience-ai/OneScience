import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timeit import default_timer
from pathlib import Path
from tqdm import tqdm

from onescience.utils.YParams import YParams
from onescience.distributed.manager import DistributedManager
from onescience.datapipes.cfd import BENODatapipe
from torch.nn.parallel import DistributedDataParallel as DDP

# 模型与工具函数
from onescience.models.beno.BE_MPNN import HeteroGNS
from onescience.utils.beno.utilities import LpLoss


def main():
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device
    
    cfg = YParams("conf/beno.yaml", "beno_config")
    
    # 输出目录设置
    output_dir = Path(cfg.training.output_dir)
    if dist.rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output: {output_dir}")

    # 数据管道初始化
    datapipe = BENODatapipe(cfg, distributed=(dist.world_size > 1))
    train_loader, train_sampler = datapipe.train_dataloader()
    test_loader, test_sampler = datapipe.test_dataloader()
    
    u_normalizer = datapipe.u_normalizer.to(device)

    # 模型构建
    model_cfg = cfg.model
    if model_cfg.act == "relu":
        activation = nn.ReLU
    elif model_cfg.act == "elu":
        activation = nn.ELU
    elif model_cfg.act == "leakyrelu":
        activation = nn.LeakyReLU
    else:
        activation = nn.SiLU
    
    model = HeteroGNS(
        nnode_in_features=model_cfg.nnode_in_features,
        nnode_out_features=model_cfg.nnode_out_features,
        nedge_in_features=model_cfg.nedge_in_features,
        nmlp_layers=model_cfg.nmlp_layers,
        activation=activation,
        boundary_dim=model_cfg.boundary_dim,
        trans_layer=model_cfg.trans_layer
    ).to(device)
    
    # 分布式并行
    if dist.world_size > 1:
        model = DDP(
            model,
            device_ids=[dist.local_rank],
            output_device=dist.local_rank
        )
        
    # 优化器与学习率调度器
    train_cfg = cfg.training
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.optimizer.lr,
        weight_decay=train_cfg.optimizer.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=train_cfg.scheduler.T_0,
        T_mult=train_cfg.scheduler.T_mult
    )
    
    myloss = LpLoss(size_average=False)
    
    if dist.rank == 0:
        print("Starting Training...")
    
    for epoch in range(train_cfg.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        model.train()
        t1 = default_timer()
        
        train_mse = 0.0
        train_l2 = 0.0
        num_batches = 0
        
        # 仅在 rank 0 显示训练进度条
        if dist.rank == 0:
            iterator = tqdm(
                train_loader,
                desc=f"Epoch {epoch}/{train_cfg.epochs}",
                dynamic_ncols=True,
                leave=False
            )
        else:
            iterator = train_loader
        
        # 训练循环
        for batch in iterator:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch)
            
            # 归一化空间上的 MSE 损失
            loss = F.mse_loss(
                out.view(-1, 1),
                batch["G1+2"].y.view(-1, 1)
            )
            loss.backward()
            optimizer.step()
            
            # 反归一化后的 L2 误差评估
            with torch.no_grad():
                pred_denorm = u_normalizer.decode(
                    out.view(batch.num_graphs, -1),
                    sample_idx=batch["G1"].sample_idx.view(batch.num_graphs, -1)
                )
                target_denorm = u_normalizer.decode(
                    batch["G1+2"].y.view(batch.num_graphs, -1),
                    sample_idx=batch["G1"].sample_idx.view(batch.num_graphs, -1)
                )
                l2 = myloss(pred_denorm, target_denorm)
            
            train_mse += loss.item()
            train_l2 += l2.item()
            num_batches += 1
            
            if dist.rank == 0:
                iterator.set_postfix({
                    "mse": f"{loss.item():.2e}",
                    "l2": f"{l2.item():.2e}"
                })
            
        scheduler.step()
        
        # 多卡同步训练指标
        if dist.world_size > 1:
            metrics_tensor = torch.tensor(
                [train_mse, train_l2, num_batches],
                device=device
            )
            torch.distributed.all_reduce(metrics_tensor)
            train_mse, train_l2, total_batches = metrics_tensor.tolist()
            train_mse /= total_batches
            train_l2 /= datapipe.train_dataset.ntrain
        else:
            train_mse /= num_batches
            train_l2 /= datapipe.train_dataset.ntrain

        # 测试阶段
        model.eval()
        test_l2 = 0.0
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch)
                
                pred_denorm = u_normalizer.decode(
                    out.view(batch.num_graphs, -1),
                    sample_idx=batch["G1"].sample_idx.view(batch.num_graphs, -1)
                )
                test_l2 += myloss(
                    pred_denorm,
                    batch["G1+2"].y.view(batch.num_graphs, -1)
                ).item()
                
        if dist.world_size > 1:
            test_l2_tensor = torch.tensor(test_l2, device=device)
            torch.distributed.all_reduce(test_l2_tensor)
            test_l2 = test_l2_tensor.item()
            
        test_l2 /= datapipe.test_dataset.ntest
        
        t2 = default_timer()
        
        if dist.rank == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"Train MSE: {train_mse:.6f} | "
                f"Train L2: {train_l2:.6f} | "
                f"Test L2: {test_l2:.6f} | "
                f"Time: {t2 - t1:.1f}s"
            )
            
            if (epoch + 1) % cfg.training.save_period == 0:
                ckpt_path = output_dir / f"model_epoch_{epoch}.pt"
                model_state = (
                    model.module.state_dict()
                    if hasattr(model, "module")
                    else model.state_dict()
                )
                torch.save(model_state, ckpt_path)

    dist.cleanup()


if __name__ == "__main__":
    main()
