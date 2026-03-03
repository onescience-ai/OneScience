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
    style='PanguDownSample2D',
    input_resolution=(181, 360),
    output_resolution=(91, 180),
    in_dim=192)
x = torch.randn(2, 65160, 192)  # (B, lat*lon, C)
out = downsample1(x)

print('Function: Pangu Down Sample 2D Forward Pass')
print(f'output shape: {out.shape}')
print( 'target shape: torch.Size([2, 16380, 384])\n')

downsample2 = OneSample(
    style='PanguDownSample2D',
    input_resolution=(128, 256),
    output_resolution=(64, 128),
    in_dim=192)
x = torch.randn(2, 32768, 192)  # (B, lat*lon, C)
out = downsample2(x)

print('Function: Pangu Down Sample 2D Forward Pass')
print(f'output shape: {out.shape}')
print( 'target shape: torch.Size([2, 8192, 384])\n')
