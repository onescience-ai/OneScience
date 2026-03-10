import torch
from onescience.modules import OneRecovery
import warnings
import math

# 忽略有关 'torch.meshgrid' 的警告
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")

recovery = OneRecovery(
    style='XihePatchRecovery',
    img_size=(721, 1440),
    patch_size=(4, 4),
    in_chans=384,
    out_chans=4,
)
x = torch.randn(2, 384, 181, 360)
out = recovery(x)

print('Function: XiHe Patch Recovery 2D Forward')
print(f'output shape: {out.shape}')
print( 'target shape: torch.Size([2, 4, 721, 1440])\n')
