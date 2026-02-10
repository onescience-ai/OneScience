import torch
from onescience.modules.resample.DownSample import DownSample2D

downsample = DownSample2D(64, (128, 256), (64, 128))
x = torch.randn(8, 32768, 64)
out = downsample(x)
print('output:  ', out.shape)
print(f'expected: torch.Size([8, 8192, 128])')