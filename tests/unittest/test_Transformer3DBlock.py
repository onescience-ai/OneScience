import torch
from onescience.modules.attention.Fuser import Transformer3DBlock
import warnings

# 忽略有关 'torch.meshgrid' 的警告
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")

# 基础用法
# input_resolution=(13, 128, 256), L=13*128*256=425984
block = Transformer3DBlock(
    dim=192,
    input_resolution=(13, 128, 256),
    num_heads=6,
    window_size=(2, 6, 12),
    shift_size=(1, 3, 6)
)
# 输入序列格式: (B, L, C)
x = torch.randn(4, 425984, 192)
out = block(x)
print('output:  ', out.shape)
print(f'expected: torch.Size([4, 425984, 192])')