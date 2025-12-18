import torch
import random
<<<<<<< HEAD
from onescience.utils.deepcfd.functions import *
from onescience.utils.deepcfd.functions import *
from torch.utils.data import TensorDataset

# 根据训练代码的保存逻辑修改推理代码
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_input = "./dataX.pkl"
model_output = "./dataY.pkl"
model_path = "./models/mymodel.pt"
=======
import os
import sys
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Imports
from onescience.utils.YParams import YParams
from onescience.distributed.manager import DistributedManager
from onescience.datapipes import DeepCFDDatapipe
from onescience.utils.deepcfd.functions import visualize # 假设此可视化函数保留
>>>>>>> recover-cfd

def init_model_from_config(model_config_dict):
    """根据保存的配置字典重建模型"""
    model_name = model_config_dict['name']
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
        
    return net_class(
        model_config_dict['in_channels'],
        model_config_dict['out_channels'],
        filters=model_config_dict['filters'],
        kernel_size=model_config_dict['kernel_size'],
        batch_norm=model_config_dict['batch_norm'],
        weight_norm=model_config_dict['weight_norm']
    )

<<<<<<< HEAD
# 从checkpoint中获取训练时的配置参数
config = checkpoint["config"]
kernel_size = config["kernel_size"]
filters = config["filters"]
arch_type = config["arch_type"]

# 动态导入正确的网络架构
if arch_type == "UNet":
    from onescience.models.deepcfd.UNet import UNet

    net_class = UNet
elif arch_type == "UNetEx":
    from onescience.models.deepcfd.UNetEx import UNetEx

    net_class = UNetEx
elif arch_type == "AutoEncoder":
    from onescience.models.deepcfd.AutoEncoder import AutoEncoder

    net_class = AutoEncoder
else:
    raise ValueError(f"error network: {arch_type}")

# 根据保存的配置重建模型
model = net_class(
    3,  # 输入通道数
    3,  # 输出通道数
    filters=filters,
    kernel_size=kernel_size,
    batch_norm=False,
    weight_norm=False,
)

# 加载模型状态
model.load_state_dict(checkpoint["model_state"])

model.to(device)
model.eval()

# 加载数据
x = pickle.load(open(model_input, "rb"))
y = pickle.load(open(model_output, "rb"))

# 准备数据
indices = list(range(len(x)))
random.shuffle(indices)
x = x[indices]
y = y[indices]

x = torch.FloatTensor(x)
y = torch.FloatTensor(y)

batch = x.shape[0]
nx = x.shape[2]
ny = x.shape[3]

channels_weights = (
    torch.sqrt(
        torch.mean(y.permute(0, 2, 3, 1).reshape((batch * nx * ny, 3)) ** 2, dim=0)
    )
    .view(1, -1, 1, 1)
    .to(device)
)

# 划分数据集
train_data, test_data = split_tensors(x, y, ratio=0.7)
train_dataset, test_dataset = TensorDataset(*train_data), TensorDataset(*test_data)
test_x, test_y = test_dataset[:]

# 推理
with torch.no_grad():
    out = model(test_x[:10].to(device))

error = torch.abs(out.cpu() - test_y[:10].cpu())
s = 0
visualize(
    test_y[:10].cpu().detach().numpy(),
    out[:10].cpu().detach().numpy(),
    error[:10].cpu().detach().numpy(),
    s,
)

print("Inference completed, visualization results saved to the 'result' folder!")
=======
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
>>>>>>> recover-cfd
