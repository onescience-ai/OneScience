import torch
from onescience.modules.patch.patch_recovery import PatchRecovery3D

patch_recovery = PatchRecovery3D(
    img_size=(13, 128, 256),
    patch_size=(1, 4, 4),
    in_chans=192,
    out_chans=5
)
x = torch.randn(4, 192, 13, 32, 64)
out = patch_recovery(x)
print('output:  ', out.shape)
print(f'expected: torch.Size([4, 5, 13, 128, 256])')