import torch
from onescience.modules import OneTransformer
import warnings

# 忽略有关 'torch.meshgrid' 的警告
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")

EarthTransformer3DBlock = OneTransformer(
    style="EarthTransformer3DBlock",
    dim=192,
    input_resolution=(13, 128, 256),
    num_heads=6,
    window_size=(2, 6, 12),
    shift_size=(0, 0, 0),   # 普通窗口
)
x = torch.randn(2, 13 * 128 * 256, 192)  # (B*num_lon, num_lat, N, C)
out = EarthTransformer3DBlock(x)
print('Function: EarthTransformer3DBlock Forward')
print(f'output shape: {out.shape}')
print( 'target shape: torch.Size([2, 425984, 192])\n')

EarthTransformer3DBlock = OneTransformer(
    style="EarthTransformer3DBlock",
    dim=192,
    input_resolution=(13, 128, 256),
    num_heads=6,
    window_size=(2, 6, 12),
    shift_size=(1, 3, 6),   # 半窗口移位
)
out = EarthTransformer3DBlock(x)
print('Function: EarthTransformer3DBlock Forward')
print(f'output shape: {out.shape}')
print( 'target shape: torch.Size([2, 425984, 192])\n')