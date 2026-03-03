import torch
from onescience.modules import OneAttention
import warnings

# 忽略有关 'torch.meshgrid' 的警告
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")

EarthAttention2D = OneAttention(
    style="EarthAttention2D",
    dim=192,
    input_resolution=(128, 256),
    window_size=(8, 8),
    num_heads=6
)
x = torch.randn(128, 16, 64, 192)  # (B*num_lon, num_lat, N, C)
out = EarthAttention2D(x)
print('Function: EarthAttention2D Forward Pass')
print(f'output shape: {out.shape}')
print( 'target shape: torch.Size([128, 16, 64, 192])\n')

# 带mask的前向传播（用于循环边界填充场景）
mask = torch.zeros(32, 16, 64, 64)  # (num_lon, num_lat, N, N)
out = EarthAttention2D(x, mask=mask)
print('Function: masked-EarthAttention2D Forward Pass')
print(f'output shape: {out.shape}')
print( 'target shape: torch.Size([128, 16, 64, 192])\n')