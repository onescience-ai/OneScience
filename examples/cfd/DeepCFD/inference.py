import os
import json
import torch
import pickle
import random
from onescience.utils.deepcfd.functions import *
from onescience.utils.deepcfd.functions import *
from torch.utils.data import TensorDataset

# 根据训练代码的保存逻辑修改推理代码
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_input = "./dataX.pkl"
model_output = "./dataY.pkl"
model_path = "./models/mymodel.pt"

# 先加载模型配置
checkpoint = torch.load(model_path, map_location=device, weights_only=True)

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
