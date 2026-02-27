import torch
from onescience.modules.patch.PatchEmbed import OnePatchEmbed
import warnings

# 忽略有关 'torch.meshgrid' 的警告
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")
# 2D patch嵌入
patch_embed = OnePatchEmbed(
    img_size=(128, 256),
    patch_size=(4, 4),
    in_chans=3,
    embed_dim=96,
    style='pangu'
)
x = torch.randn(8, 3, 128, 256)
out = patch_embed(x)
print('Function: 2D PatchEmbed')
print(f'output shape: {out.shape}')
print( 'target shape: torch.Size([8, 96, 32, 64])\n')

# 使用LayerNorm
patch_embed = OnePatchEmbed(
    img_size=(128, 256),
    patch_size=(4, 4),
    in_chans=3,
    embed_dim=96,
    norm_layer=torch.nn.LayerNorm,
    style='pangu'
)
x = torch.randn(8, 3, 128, 256)
out = patch_embed(x)
print('Function: 2D PatchEmbed with LayerNorm')
print(f'output shape: {out.shape}')
print( 'target shape: torch.Size([8, 96, 32, 64])\n')

# 3D patch嵌入
patch_embed = OnePatchEmbed(
    img_size=(13, 128, 256),
    patch_size=(1, 4, 4),
    in_chans=5,
    embed_dim=192,
    style='pangu'
)
x = torch.randn(4, 5, 13, 128, 256)
out = patch_embed(x)
print('Function: 3D PatchEmbed')
print(f'output shape: {out.shape}')
print( 'target shape: torch.Size([4, 192, 13, 32, 64])\n')


