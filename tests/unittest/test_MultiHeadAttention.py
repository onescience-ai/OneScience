import torch
from onescience.modules.attention.MultiHeadAttention import MultiHeadAttention
attn = MultiHeadAttention(dim=128, heads=8, dim_head=16)
x = torch.randn(8, 100, 128)
out = attn(x)
out.shape
print('=======test:MultiHeadAttention=======')
print('output:  ', out.shape)
print(f'expected: torch.Size([8, 100, 128])')
