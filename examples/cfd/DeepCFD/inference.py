import torch
import random
import os
import sys
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Imports
from onescience.utils.YParams import YParams
from onescience.distributed.manager import DistributedManager
from onescience.datapipes.cfd import DeepCFDDatapipe
from onescience.utils.deepcfd.functions import visualize # 假设此可视化函数保留

def init_model_from_config(model_config_dict):
    """根据保存的配置字典重建模型，需与训练时的参数名匹配"""
    model_name = model_config_dict['name']
    
    if model_name == "UNet":
        from onescience.models.deepcfd.UNet import UNet
        net_class = UNet
    elif model_name == "UNetEx":
        from onescience.models.deepcfd.UNetEx import UNetEx
        net_class = UNetEx
    else:
        raise ValueError(f"Unknown network: {model_name}")
    
    return net_class(
        in_channels=model_config_dict['in_channels'],
        out_channels=model_config_dict['out_channels'],
        base_channels=model_config_dict['base_channels'], 
        num_stages=model_config_dict['num_stages'],       
        bilinear=model_config_dict.get('bilinear', True), 
        normtype=model_config_dict.get('normtype', 'none'),
        kernel_size=model_config_dict['kernel_size']
    )

def main():
    # 1. Init
    # 推理通常单卡即可
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device
    
    # 2. Config
    config_path = "conf/deepcfd.yaml"
    cfg = YParams(config_path, "root")
    
    # 3. Load Checkpoint
    output_dir = Path(cfg.training.output_dir)
    model_path = output_dir / "best_model.pt"
    
    if dist.rank == 0:
        print(f"Loading checkpoint from {model_path}")
        
    if not model_path.exists():
        print(f"Error: Checkpoint not found at {model_path}")
        return

    checkpoint = torch.load(model_path, map_location=device)
    
    # 4. Rebuild Model
    # 使用 checkpoint 中保存的 config 来确保架构一致，或者使用 yaml 配置
    saved_model_config = checkpoint.get("config", cfg.model.to_dict())
    model = init_model_from_config(saved_model_config)
    
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    
    # 5. Data
    # 使用 Datapipe 获取测试数据
    datapipe = DeepCFDDatapipe(cfg.datapipe, distributed=False)
    test_loader, _ = datapipe.test_dataloader()
    
    # 6. Inference & Visualize
    if dist.rank == 0:
        print("Running Inference on first batch...")
        
        # 获取一个 batch
        batch = next(iter(test_loader))
        x = batch['x'].to(device)
        y = batch['y'].to(device)
        
        with torch.no_grad():
            out = model(x)
            
        # Error calculation
        error = torch.abs(out.cpu() - y.cpu())
        
        # Convert to numpy for visualization
        y_np = y.cpu().numpy()
        out_np = out.cpu().numpy()
        err_np = error.cpu().numpy()
        
        num_vis = min(5, x.shape[0])
        print(f"Visualizing {num_vis} samples...")
        
        vis_dir = output_dir / "vis_results"
        vis_dir.mkdir(exist_ok=True)
        
        for i in range(num_vis):
            print(f"Plotting sample {i}")
            # 这里的调用方式需参考原始 functions.py
            visualize(
                y_np,    # 传入完整 batch
                out_np,  # 传入完整 batch
                err_np,  # 传入完整 batch
                i,        # 索引，函数内部用它来做切片 sample_y[s]
                save_dir=str(vis_dir)
            )        
        print(f"Visualization saved to {vis_dir}")

    dist.cleanup()

if __name__ == "__main__":
    main()