import torch
from onescience.modules.resample.ReSample import OneReSample
import warnings

# 忽略有关 'torch.meshgrid' 的警告
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")
# 2D上采样 - 对应原UpSample2D
resample = OneReSample(128, 64, (64, 128), (128, 256), style='pangu')
x = torch.randn(8, 8192, 128)
out = resample(x)
print('Function: 2D Upsample')
print(f'output shape: {out.shape}')
print( 'target shape: torch.Size([8, 32768, 64])\n')

# 3D上采样 - 对应原UpSample3D
resample = OneReSample(256, 128, (13, 32, 64), (13, 64, 128), style='pangu')
x = torch.randn(4, 26624, 256)
out = resample(x)
print('Function: 3D Upsample')
print(f'output shape: {out.shape}')
print( 'target shape: torch.Size([4, 106496, 128])\n')


# 2D下采样 - 对应原DownSample2D
resample = OneReSample(64, 128, (128, 256), (64, 128), style='pangu')
x = torch.randn(8, 32768, 64)
out = resample(x)
print('Function: 2D Downsample')
print(f'output shape: {out.shape}')
print( 'target shape: torch.Size([8, 8192, 128])\n')


# 3D下采样 - 对应原DownSample3D
resample = OneReSample(128, 256, (13, 64, 128), (13, 32, 64), style='pangu')
x = torch.randn(4, 106496, 128)
out = resample(x)
print('Function: 3D Downsample')
print(f'output shape: {out.shape}')
print( 'target shape: torch.Size([4, 26624, 256])\n')
