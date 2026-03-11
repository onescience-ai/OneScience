import torch
import numpy as np
from onescience.modules.func_utils.xihe_utils import change_mask
from onescience.modules import OneTransformer
import warnings

# 忽略有关 'torch.meshgrid' 的警告
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")

XihelocalTransformer = OneTransformer(
    style="XihelocalTransformer",
    dim=192,
    input_resolution=(1, 341, 360),
    num_heads=6,
)
x = torch.randn(1, 341*360, 192)  
mask = np.load('/root/private_data/hanym/modules/onescience/src/onescience/models/xihe/20210628_zos_ocean_mask.npy')
mask1=change_mask(mask,x,341,360)

obj={
    "x":x,
    "mask":mask1
}

out = XihelocalTransformer(obj)
print('Function: XihelocalTransformer Forward')
print(f'output shape: {out.shape}')
print( 'target shape: torch.Size([1, 122760, 192])\n')
