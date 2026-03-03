import torch
from onescience.modules import OneFuser
import warnings
import math

# 忽略有关 'torch.meshgrid' 的警告
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")
dim = 256
input_resolution = (10, 181, 360)
fuser = OneFuser(
    style='PanguFuser',
    input_resolution=input_resolution,
    dim=dim,
    depth=4,
    num_heads=8,
    window_size=(2, 6, 12),
)
B, T, H, W = 2, 10, 181, 360
x = torch.randn(B, T * H * W, dim)
out = fuser(x)

print('Function: FengWuFuser Forward')
print(f'output shape: {out.shape}')
print( 'target shape: torch.Size([2, 651600, 256]))\n')
