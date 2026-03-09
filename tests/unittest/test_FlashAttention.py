import torch
from onescience.modules.attention.FlashAttention import FlashAttention
attn = FlashAttention(dim=64, heads=8, dim_head=8)
x = torch.randn(8, 128, 64)
out = attn(x)
print('=======test:FlashAttention=======')
print('output:  ', out.shape)
print(f'expected: torch.Size([8, 100, 64])')