import torch
from onescience.modules.attention.SelfAttention import SelfAttention

sa = SelfAttention(dim=64, heads=8)
x = torch.randn(8, 256, 64)
mask = torch.ones(8, 256).bool()
out = sa(x, mask=mask)
print('=======test:SelfAttention=======')
print('output:  ', out.shape)
print(f'expected: torch.Size([8, 256, 64])')