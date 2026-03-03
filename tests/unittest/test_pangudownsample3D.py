import torch
from onescience.modules import OneSample
import warnings
import math

# 忽略有关 'torch.meshgrid' 的警告
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")

# 气象场分辨率 181×360 → 91×180 下采样
# h_pad = 91*2 - 181 = 1（底部补1行）
# w_pad = 180*2 - 360 = 0（无需补齐）
# 输入 token 数: 181 * 360 = 65160
# 输出 token 数:  91 * 180 = 16380
downsample1 = OneSample(
    style='PanguDownSample3D',
    input_resolution=(8, 181, 360),
    output_resolution=(8, 91, 180),
    in_dim=192)
x = torch.randn(2, 521280, 192)  # (B, lat*lon, C)
out = downsample1(x)

print('Function: Pangu Down Sample 3D Forward Pass')
print(f'output shape: {out.shape}')
print( 'target shape: torch.Size([2, 131040, 384])\n')

downsample2 = OneSample(
    style='PanguDownSample3D',
    input_resolution=(13, 128, 256),
    output_resolution=(13, 64, 128),
    in_dim=192)
x = torch.randn(2, 425984, 192)  # (B, lat*lon, C)
out = downsample2(x)

print('Function: Pangu Down Sample 3D Forward Pass')
print(f'output shape: {out.shape}')
print( 'target shape: torch.Size([2, 106496, 384])\n')
