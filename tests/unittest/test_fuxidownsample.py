import torch
from onescience.modules import OneSample
import warnings
import math

# 忽略有关 'torch.meshgrid' 的警告
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")

down = OneSample(
    style='FuxiDownSample',
    in_chans=1536, 
    out_chans=1536, 
    num_groups=32, 
    num_residuals=2)
x = torch.randn(2, 1536, 180, 360)
out = down(x)

print('Function: Fuxi Down Sample Forward Pass')
print(f'output shape: {out.shape}')
print( 'target shape: torch.Size([2, 1536, 90, 180])\n')
