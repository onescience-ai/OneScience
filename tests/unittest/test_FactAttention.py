import torch
from onescience.modules.attention.FactAttention import FactAttention2D, FactAttention3D
fact_attn = FactAttention2D(dim=128, heads=4, dim_head=32, shapelist=(32, 32))
x = torch.randn(2, 32*32, 128)
out = fact_attn(x)
print('=======test:FactAttention2D=======')
print('output:  ', out.shape)
print(f'expected: torch.Size([2, 1024, 128])')

fact_attn = FactAttention3D(dim=512, heads=8, dim_head=64, shapelist=(16, 16, 16))
x = torch.randn(2, 16**3, 512)
out = fact_attn(x)
print('=======test:FactAttention2D=======')
print('output:  ', out.shape)
print(f'expected: torch.Size([2, 4096, 512])')