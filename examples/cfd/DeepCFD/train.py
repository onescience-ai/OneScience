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
<<<<<<< HEAD
from torch.nn.parallel import DistributedDataParallel


def generate_metrics_list(metrics_def):
    list = {}
    for name in metrics_def.keys():
        list[name] = []
    return list
=======
from onescience.datapipes import DeepCFDDatapipe
from torch.nn.parallel import DistributedDataParallel as DDP
>>>>>>> recover-cfd

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
    # 注意：原始代码 reshape 有点复杂，这里简化逻辑但保持数学等价
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
            
    # DDP Reduce logic could be added here for exact metrics across all GPUs
    # For now, we return metrics from local process (or avg if we implemented AllReduce)
    
    avg_loss = total_loss / num_batches # 注意：这是 sum loss，可能很大
    # 原始代码 metrics 计算比较简单，这里返回 sum
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
<<<<<<< HEAD
        # 只在主进程计算total_loss
        if dist.rank == 0:
            total_loss += loss.item()
        scope["batch"] = tensors
        scope["loss"] = loss
        scope["output"] = output
        scope["batch_metrics"] = {}
        for name, metric in metrics_def.items():
            value = metric["on_batch"](scope)
            scope["batch_metrics"][name] = value
            metrics_list[name].append(value)
        if on_batch is not None:
            on_batch(scope)
    scope["metrics_list"] = metrics_list
    metrics = {}
    if dist.rank == 0:
        for name in metrics_def.keys():
            scope["list"] = scope["metrics_list"][name]
            metrics[name] = metrics_def[name]["on_epoch"](scope)
    return total_loss, metrics


def train(
    scope,
    train_dataset,
    val_dataset,
    device,
    patience=10,
    batch_size=256,
    print_function=print,
    eval_model=None,
    on_train_batch=None,
    on_val_batch=None,
    on_train_epoch=None,
    on_val_epoch=None,
    after_epoch=None,
):
    dist = DistributedManager()
    early_stopping = EarlyStopping(
        patience,
        verbose=True,
        checkpoint_dir="checkpoints",
        is_ddp=scope.get("ddp", False),  # 从训练作用域获取 DDP 状态
    )

    epochs = scope["epochs"]
    model = scope["model"]
    metrics_def = scope["metrics_def"]
    scope = copy.copy(scope)

    scope["best_train_metric"] = None
    scope["best_train_loss"] = float("inf")
    scope["best_val_metrics"] = None
    scope["best_val_loss"] = float("inf")
    scope["best_model"] = None

    # 创建分布式采样器
    train_sampler = (
        DistributedSampler(train_dataset) if scope.get("ddp", False) else None
    )
    val_sampler = (
        DistributedSampler(val_dataset, shuffle=False)
        if scope.get("ddp", False)
        else None
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, sampler=val_sampler, shuffle=False
    )
    skips = 0
    for epoch_id in range(1, epochs + 1):
        scope["epoch"] = epoch_id
        # 设置epoch给分布式采样器
        if scope.get("ddp", False) and train_sampler is not None:
            train_loader.sampler.set_epoch(epoch_id)
        if dist.rank == 0:
            print_function("Epoch #" + str(epoch_id), flush=True)

        # Training
        scope["dataset"] = train_dataset
        train_loss, train_metrics = epoch(
            scope, train_loader, on_train_batch, training=True
        )
        if dist.rank == 0:
            scope["train_loss"] = train_loss
            scope["train_metrics"] = train_metrics
            print_function("\tTrain Loss = " + str(train_loss), flush=True)
            for name in metrics_def.keys():
                print_function(
                    "\tTrain "
                    + metrics_def[name]["name"]
                    + " = "
                    + str(train_metrics[name]),
                    flush=True,
                )
            if on_train_epoch is not None:
                on_train_epoch(scope)
        del scope["dataset"]

        # Validation
        scope["dataset"] = val_dataset
        with torch.no_grad():
            val_loss, val_metrics = epoch(
                scope, val_loader, on_val_batch, training=False
            )
        if dist.rank == 0:
            scope["val_loss"] = val_loss
            scope["val_metrics"] = val_metrics
            print_function("\tValidation Loss = " + str(val_loss), flush=True)
            for name in metrics_def.keys():
                print_function(
                    "\tValidation "
                    + metrics_def[name]["name"]
                    + " = "
                    + str(val_metrics[name]),
                    flush=True,
                )
            if on_val_epoch is not None:
                on_val_epoch(scope)
        del scope["dataset"]

        # Selection
        stop_flag = False
        if dist.rank == 0:
            is_best = None
            if eval_model is not None:
                is_best = eval_model(scope)
            if is_best is None:
                is_best = val_loss < scope["best_val_loss"]
            if is_best:
                scope["best_train_metric"] = train_metrics
                scope["best_train_loss"] = train_loss
                scope["best_val_metrics"] = val_metrics
                scope["best_val_loss"] = val_loss
                scope["best_model"] = copy.deepcopy(model)
                print_function("Model saved!", flush=True)
                skips = 0
            else:
                skips += 1
            if after_epoch is not None:
                after_epoch(scope)

            # 只在主进程更新早停
            early_stopping(val_loss, scope["best_model"])
            if early_stopping.early_stop:
                print_function("Early stopping", flush=True)
                stop_flag = True

        # 广播停止标志 - 仅当在分布式环境中
        if dist.world_size > 1:
            # 确保所有进程知道是否停止
            stop_tensor = torch.tensor([stop_flag], dtype=torch.bool, device=device)
            torch.distributed.broadcast(stop_tensor, src=0)
            stop_flag = stop_tensor.item()

        # 检查是否提前停止（所有进程都会检查）
        if stop_flag:
            break

    # 只在主进程返回
    if dist.rank == 0:
        return (
            scope["best_model"],
            scope["best_train_metric"],
            scope["best_train_loss"],
            scope["best_val_metrics"],
            scope["best_val_loss"],
        )
    else:
        return None, None, None, None, None
=======
            
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
                        "config": cfg.model.to_dict(), # 保存模型配置以便推理重建
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
>>>>>>> recover-cfd

    dist.cleanup()

<<<<<<< HEAD
def train_model(
    model,
    loss_func,
    train_dataset,
    val_dataset,
    optimizer,
    process_batch=None,
    eval_model=None,
    on_train_batch=None,
    on_val_batch=None,
    on_train_epoch=None,
    on_val_epoch=None,
    after_epoch=None,
    epochs=100,
    batch_size=256,
    patience=10,
    device=0,
    **kwargs
):
    dist_manager = DistributedManager()
    scope = {}
    # DDP封装
    model = model.to(device)
    if dist_manager.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist_manager.local_rank],
            output_device=dist_manager.local_rank,
        )
        scope = {"ddp": True}
    else:
        scope = {"ddp": False}
    scope["model"] = model
    scope["loss_func"] = loss_func
    scope["train_dataset"] = train_dataset
    scope["val_dataset"] = val_dataset
    scope["optimizer"] = optimizer
    scope["process_batch"] = process_batch
    scope["epochs"] = epochs
    scope["batch_size"] = batch_size
    scope["device"] = device
    metrics_def = {}
    names = []
    for key in kwargs.keys():
        parts = key.split("_")
        if len(parts) == 3 and parts[0] == "m":
            if parts[1] not in names:
                names.append(parts[1])
    for name in names:
        if (
            "m_" + name + "_name" in kwargs
            and "m_" + name + "_on_batch" in kwargs
            and "m_" + name + "_on_epoch" in kwargs
        ):
            metrics_def[name] = {
                "name": kwargs["m_" + name + "_name"],
                "on_batch": kwargs["m_" + name + "_on_batch"],
                "on_epoch": kwargs["m_" + name + "_on_epoch"],
            }
        else:
            print("Warning: " + name + " metric is incomplete!")
    scope["metrics_def"] = metrics_def
    return train(
        scope,
        train_dataset,
        val_dataset,
        device,
        eval_model=eval_model,
        on_train_batch=on_train_batch,
        on_val_batch=on_val_batch,
        on_train_epoch=on_train_epoch,
        on_val_epoch=on_val_epoch,
        after_epoch=after_epoch,
        batch_size=batch_size,
        patience=patience,
    )
=======
if __name__ == "__main__":
    main()
>>>>>>> recover-cfd
