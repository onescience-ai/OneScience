import torch
from onescience.modules.equivariant.group_conv import GConv2d, GConv3d
# 1. 第一层 (Lifting)
gconv = GConv2d(32, 64, kernel_size=3, first_layer=True)
x = torch.randn(2, 32, 64, 64)
out = gconv(x)
print('=======test:GSpectralConv2d(第一层)=======')
print('output:  ', out.shape)
print(f'expected: torch.Size([2, 128, 64, 64])')

# 1. 第一层 (Lifting)
gconv2 = GConv2d(64, 64, kernel_size=3)
out2 = gconv2(out)
print('=======test:GSpectralConv2d(中间层)=======')
print('output:  ', out2.shape)
print(f'expected: torch.Size([2, 256, 64, 64])')