import torch
from onescience.modules import OneEmbedding
import warnings
import math

# 忽略有关 'torch.meshgrid' 的警告
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")
patch_embed = OneEmbedding(
    style=('PanguEmbedding3D'),
    img_size=(13, 128, 256),
    patch_size=(1, 4, 4),
    in_chans=5,
    embed_dim=192
)
x = torch.randn(4, 5, 13, 128, 256)
out = patch_embed(x)

print('Function: Pangu Embedding 3D Forward Pass')
print(f'output shape: {out.shape}')
print( 'target shape: torch.Size([4, 192, 13, 32, 64])\n')
