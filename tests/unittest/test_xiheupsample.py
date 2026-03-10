import torch
from onescience.modules import OneSample
import warnings
import math

# 忽略有关 'torch.meshgrid' 的警告
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")

up = OneSample(
    style='XiheUpSample',
    in_dim=384, 
    out_dim=192, 
    input_resolution=(64, 128),
    output_resolution=(128, 256),)
x = torch.randn(2, 8192, 384)
out = up(x)

print('Function: XiHe up Sample Forward')
print(f'output shape: {out.shape}')
print( 'target shape: torch.Size([2, 32768, 192])\n')
