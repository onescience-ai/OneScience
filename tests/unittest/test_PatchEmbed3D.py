import torch
from onescience.modules.patch.patch_embed import PatchEmbed3D

patch_embed = PatchEmbed3D(
    img_size=(13, 128, 256),
    patch_size=(1, 4, 4),
    in_chans=5,
    embed_dim=192
)
x = torch.randn(4, 5, 13, 128, 256)
out = patch_embed(x)
print('output:  ', out.shape)
print(f'expected: torch.Size([4, 192, 13, 32, 64])')