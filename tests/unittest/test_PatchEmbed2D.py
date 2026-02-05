import torch
from onescience.modules.patch.patch_embed import PatchEmbed2D

patch_embed = PatchEmbed2D(
    img_size=(128, 256),
    patch_size=(4, 4),
    in_chans=3,
    embed_dim=96
)
x = torch.randn(8, 3, 128, 256)
out = patch_embed(x)
print('output:  ', out.shape)s
print(f'expected: torch.Size([8, 96, 32, 64])')