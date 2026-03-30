import torch
from onescience.modules import OneEmbedding
import warnings

# 忽略有关 'torch.meshgrid' 的警告
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")

# Pangu-Weather surface 分支
surface_patch_embed = OneEmbedding(
    style="PanguEmbedding",
    img_size=(721, 1440),
    patch_size=(4, 4),
    in_chans=7,
    embed_dim=192
)
surface_x = torch.randn(2, 7, 721, 1440)
surface_out = surface_patch_embed(surface_x)
surface_target_shape = torch.Size([2, 192, 181, 360])

print('Function: Pangu Embedding Surface Forward')
print(f'output shape: {surface_out.shape}')
print(f'target shape: {surface_target_shape}\n')
assert surface_out.shape == surface_target_shape


# Pangu-Weather upper-air 分支
upper_air_patch_embed = OneEmbedding(
    style="PanguEmbedding",
    img_size=(13, 721, 1440),
    patch_size=(2, 4, 4),
    in_chans=5,
    embed_dim=192
)
upper_air_x = torch.randn(2, 5, 13, 721, 1440)
upper_air_out = upper_air_patch_embed(upper_air_x)
upper_air_target_shape = torch.Size([2, 192, 7, 181, 360])

print('Function: Pangu Embedding Upper Air Forward')
print(f'output shape: {upper_air_out.shape}')
print(f'target shape: {upper_air_target_shape}\n')
assert upper_air_out.shape == upper_air_target_shape
