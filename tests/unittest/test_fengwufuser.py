import torch
from onescience.modules import OneFuser
import warnings
import math

# 忽略有关 'torch.meshgrid' 的警告
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")
fuser = OneFuser(
    style='FengWuFuser',
    input_resolution=(6, 91, 180),
    dim=192 * 2,
    depth=6,
    num_heads=12,
    window_size=(2, 6, 12),
)
B, T, H, W, C = 2, 6, 91, 180, 192 * 2
x = torch.randn(B, T * H * W, C)  # 已展平的三维网格特征
out = fuser(x)

print('Function: FengWuFuser Forward Pass')
print(f'output shape: {out.shape}')
print( 'target shape: torch.Size([2, 98280, 384]))\n')
