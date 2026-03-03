import torch
from onescience.modules import OneTransformer
import warnings

# 忽略有关 'torch.meshgrid' 的警告
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")

EarthTransformer2DBlock = OneTransformer(
    style="EarthTransformer2DBlock",
    dim=192,
    input_resolution=(181, 360),
    num_heads=6,
    window_size=(6, 12),
    shift_size=(0, 0)
)
x = torch.randn(2, 181 * 360, 192)  # (B*num_lon, num_lat, N, C)
out = EarthTransformer2DBlock(x)
print('Function: EarthTransformer2DBlock Forward Pass')
print(f'output shape: {out.shape}')
print( 'target shape: torch.Size([2, 65160, 192])\n')

EarthTransformer2DBlock = OneTransformer(
    style="EarthTransformer2DBlock",
    dim=192,
    input_resolution=(181, 360),
    num_heads=6,
    window_size=(6, 12),
    shift_size=(3, 6),  # 半窗口移位
)
out = EarthTransformer2DBlock(x)
print('Function: EarthTransformer2DBlock Forward Pass')
print(f'output shape: {out.shape}')
print( 'target shape: torch.Size([2, 65160, 192])\n')