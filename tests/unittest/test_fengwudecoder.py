import torch
from onescience.modules import OneDecoder
import warnings
import math

# 忽略有关 'torch.meshgrid' 的警告
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")

B = 2
decoder = OneDecoder(style="FengWuDecoder")
x    = torch.randn(B, 16380, 384)   # (B, middle_lat*middle_lon, dim*2)
skip = torch.randn(B, 181, 360, 192) # (B, output_lat, output_lon, dim)
out = decoder([x, skip])
print('Function: FengWuDecoder Forward')
print(f'output shape: {out.shape}')
print( 'target shape: torch.Size([2, 37, 721, 1440])\n')
