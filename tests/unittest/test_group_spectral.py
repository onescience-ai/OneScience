import torch
from onescience.modules.fourier.group_spectral import GSpectralConv2d, GSpectralConv3d
gspec = GSpectralConv2d(in_channels=32, out_channels=32, modes=12, reflection=False)
x = torch.randn(2, 128, 64, 64)
out = gspec(x)
print('=======test:GSpectralConv2d=======')
print('output:  ', out.shape)
print(f'expected: torch.Size([2, 128, 64, 64])')

gspec3d = GSpectralConv3d(16, 16, modes=(8, 8, 8))
x = torch.randn(2, 64, 32, 32, 32)
out = gspec3d(x)
print('=======test:GSpectralConv3d=======')
print('output:  ', out.shape)
print(f'expected: torch.Size([2, 64, 32, 32, 32])')