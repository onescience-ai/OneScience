import torch
from onescience.modules import OneEmbedding
import warnings
import math

# 忽略有关 'torch.meshgrid' 的警告
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")
# 典型 FuXi 配置：2帧输入，70个气象变量
# patch_size=(2,4,4)，时间维度完全合并
# nT   = 2 // 2 = 1
# nLat = 721 // 4 = 180
# nLon = 1440 // 4 = 360
embedding = OneEmbedding(
    style=('FuxiEmbedding'),
    img_size=(2, 721, 1440),
    patch_size=(2, 4, 4),
    in_chans=70,
    embed_dim=1536,
)
x = torch.randn(2, 70, 2, 721, 1440)  # (B, C, T, lat, lon)
out = embedding(x)

print('Function: Fuxi Embedding Forward Pass')
print(f'output shape: {out.shape}')
print( 'target shape: torch.Size([2, 1536, 1, 180, 360])\n')
