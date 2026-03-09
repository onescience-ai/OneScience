import torch
from onescience.modules.attention.LinearAttention import MultiHeadAttention

attn = MultiHeadAttention(dim=128, heads=8, dim_head=16)
x = torch.randn(8, 100, 128)
mask = torch.ones(8, 100).bool() #创建掩码，假设后50个token是padding
mask[:, 50:] = False
out = attn(x, mask=mask)
print('=======test:Vanilla_Linear_Attention=======')
print('output:  ', out.shape)
print(f'expected: torch.Size([8, 100, 128])')

l_attn = LinearAttention(dim=64, heads=8, dim_head=8, attn_type='l2')
x = torch.randn(4, 512, 64)
out = l_attn(x)
print('=======test:LinearAttention=======')
print('output:  ', out.shape)
print(f'expected: torch.Size([4, 512, 64])')