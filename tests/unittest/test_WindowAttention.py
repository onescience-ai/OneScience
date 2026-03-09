import torch
from onescience.modules.attention.WindowAttention import WindowAttention
attn = WindowAttention(dim = 96, window_size = (7, 7), num_heads = 3)
x = torch.randn(1 * 4, 49, 96)
out = attn(x)
print('=======test:WindowAttention=======')
print('output:  ', out.shape)
print(f'expected: torch.Size([4, 49, 96])')