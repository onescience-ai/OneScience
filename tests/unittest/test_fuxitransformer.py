import torch
from onescience.modules import OneTransformer
import warnings

# 忽略有关 'torch.meshgrid' 的警告
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")

FuxiTransformer = OneTransformer(
    style="FuxiTransformer",
    embed_dim=256,
    num_groups=32,
    input_resolution=(90, 180),
    num_heads=8,
    window_size=7,
    depth=48
)
x = torch.randn(2, 256, 180, 360)  # (B*num_lon, num_lat, N, C)
out = FuxiTransformer(x)
print('Function: FuxiTransformer Forward')
print(f'output shape: {out.shape}')
print( 'target shape: torch.Size([2, 256, 180, 360])\n')