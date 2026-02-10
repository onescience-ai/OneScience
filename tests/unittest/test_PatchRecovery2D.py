import torch
from onescience.modules.patch.patch_recovery import PatchRecovery2D

patch_recovery = PatchRecovery2D(
    img_size=(128, 256),
    patch_size=(4, 4),
    in_chans=96,
    out_chans=3
)
x = torch.randn(8, 96, 32, 64)
out = patch_recovery(x)
print('output:  ', out.shape)
print(f'expected: torch.Size([8, 3, 128, 256])')