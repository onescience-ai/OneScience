import torch
from onescience.modules import OneEmbedding
import warnings
import math

# 忽略有关 'torch.meshgrid' 的警告
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")
patch_embed = OneEmbedding(
    style=('PanguEmbedding2D'),
    img_size=(128, 256),
    patch_size=(4, 4),
    in_chans=3,
    embed_dim=96
)
x = torch.randn(8, 3, 128, 256)
out = patch_embed(x)

print('Function: Pangu Embedding 2D Forward Pass')
print(f'output shape: {out.shape}')
print( 'target shape: torch.Size([8, 96, 32, 64])\n')
