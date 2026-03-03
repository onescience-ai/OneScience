import torch
from onescience.modules import OneEncoder
import warnings
import math

# 忽略有关 'torch.meshgrid' 的警告
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")

B = 2
decoder = OneEncoder(style="FengWuEncoder")
x = torch.randn(2, 37, 721, 1440)  # (B, middle_lat*middle_lon, dim*2)
out, skip = decoder(x)
print('Function: FengWuEncoder Forward Pass')
print(f'output shape: {out.shape}, skip shape: {skip.shape}')
print( 'target shape: torch.Size([2, 16380, 384]), torch.Size([2, 181, 360, 192])\n')
