import torch
from onescience.modules import OneEmbedding
import warnings
import math

# 忽略有关 'torch.meshgrid' 的警告
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")

Batch = 2
Variables = 7
img_size = [721, 1440]
patch_size = (4, 4)
embed_dim = 192

# Pangu-Weather surface 分支
surface_patch_embed = OneEmbedding(
    style="PanguEmbedding",
    img_size=img_size,
    patch_size=patch_size,
    Variables=Variables,
    embed_dim=embed_dim
).cuda()

surface_x = torch.randn(Batch, Variables, *img_size).cuda()

surface_out = surface_patch_embed(surface_x)

surface_target_shape = torch.Size(
    [
        Batch, 
        embed_dim, 
        math.ceil(img_size[0] / patch_size[0]), 
        math.ceil(img_size[1] / patch_size[1])
    ]
)

print('Function: Pangu Embedding Surface Forward')
print(f'output shape: {surface_out.shape}')
print(f'target shape: {surface_target_shape}')

if surface_out.shape == surface_target_shape:
    print('Unit test Pass\n')
else:
    print('Unit test not pass\n')


Variables = 5
img_size = [13, 721, 1440]
patch_size = (2, 4, 4)
embed_dim = 192
# Pangu-Weather upper-air 分支
upper_air_patch_embed = OneEmbedding(
    style="PanguEmbedding",
    img_size=img_size,
    patch_size=patch_size,
    Variables=Variables,
    embed_dim=embed_dim
).cuda()
upper_air_x = torch.randn(Batch, Variables, *img_size).cuda()

upper_air_out = upper_air_patch_embed(upper_air_x)

upper_air_target_shape = torch.Size(
    [
        Batch, 
        embed_dim, 
        math.ceil(img_size[0] / patch_size[0]), 
        math.ceil(img_size[1] / patch_size[1]),
        math.ceil(img_size[2] / patch_size[2])
    ]
)

print('Function: Pangu Embedding Upper Air Forward')
print(f'output shape: {upper_air_out.shape}')
print(f'target shape: {upper_air_target_shape}')
if upper_air_out.shape == upper_air_target_shape:
    print('Unit test Pass\n')
else:
    print('Unit test not pass\n')