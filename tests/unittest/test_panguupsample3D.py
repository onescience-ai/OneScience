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
upsample1 = OneSample(
    style='PanguUpSample3D',
    in_dim=384,
    out_dim=192,
    input_resolution=(8, 91, 180),
    output_resolution=(8, 181, 360))
x = torch.randn(2, 131040, 384)  # (B, lat*lon, C)
out = upsample1(x)

print('Function: Pangu Up Sample 3D Forward')
print(f'output shape: {out.shape}')
print( 'target shape: torch.Size([2, 521280, 192])\n')

upsample2 = OneSample(
    style='PanguUpSample3D',
    in_dim=384,
    out_dim=192,
    input_resolution=(13, 64, 128),
    output_resolution=(13, 128, 256))
x = torch.randn(2, 106496, 384)  # (B, lat*lon, C)
out = upsample2(x)

print('Function: Pangu Up Sample 3D Forward')
print(f'output shape: {out.shape}')
print( 'target shape: torch.Size([2, 425984, 192])\n')
