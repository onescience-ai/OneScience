import torch
from onescience.modules import OneMlp
import warnings
import math

# 忽略有关 'torch.meshgrid' 的警告
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")

mlp = OneMlp(
    style='XiheMlp',
    dim=192, 
    num_groups=32, 
    mlp_ratio=4.0)
x = torch.randn(2, 32, 192)
out = mlp(x)

print('Function: XiHe MLp Forward')
print(f'output shape: {out.shape}')
print( 'target shape: torch.Size([2, 32, 192])\n')
