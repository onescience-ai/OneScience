import torch
from onescience.modules.resample.DownSample import DownSample3D

downsample = DownSample3D(128, (13, 128, 256), (13, 64, 128))
x = torch.randn(4, 425984, 128)
out = downsample(x)
print('output:  ', out.shape)
print(f'expected: torch.Size([4, 106496, 256])')