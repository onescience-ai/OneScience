import torch
import torch.nn as nn
import numpy as np
import warnings
from onescience.modules.func_utils.xihe_utils import change_mask
from onescience.modules.attention.oneattention import OneAttention


# 忽略有关 'torch.meshgrid' 的警告
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")

FeatureGroupingAttention = OneAttention(
    style="FeatureGroupingAttention",
    dim=192,
)
x = torch.randn(1, 341*360, 192)  
mask = np.load('/root/private_data/hanym/modules/onescience/src/onescience/models/xihe/20210628_zos_ocean_mask.npy')
mask1=change_mask(mask,x,341,360)

obj={
    "x":x,
    "mask":mask1,
    "y":x
}

out = FeatureGroupingAttention(obj)
print('Function: FeatureGroupingAttention Forward')
print(f'output shape: {out.shape}')
print( 'target shape: torch.Size([1, 32, 192])\n')
