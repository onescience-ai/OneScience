import torch
from onescience.modules import OneAttention
import warnings
import math

# 忽略有关 'torch.meshgrid' 的警告
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")


EarthAttention3D = OneAttention(
    style="EarthAttention3D",
    dim=192,
    input_resolution=(14, 128, 256),
    window_size=(2, 8, 8),
    num_heads=6,
)
x = torch.randn(128, 112, 128, 192)  # (B*num_lon, num_pl*num_lat, N, C)
out = EarthAttention3D(x)
print('Function: EarthAttention3D Forward Pass')
print(f'output shape: {out.shape}')
print( 'target shape: torch.Size([128, 112, 128, 192])\n')

# 带mask的前向传播（用于循环边界填充场景）
mask = torch.zeros(32, 112, 128, 128)  # (num_lon, num_lat, N, N)
out = EarthAttention3D(x, mask=mask)
print('Function: masked-EarthAttention3D Forward Pass')
print(f'output shape: {out.shape}')
print( 'target shape: torch.Size([128, 112, 128, 192])\n')
