import torch
from onescience.modules.attention.Fuser import FuserLayer
import warnings

# 忽略有关 'torch.meshgrid' 的警告
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")

# 2 层 Transformer
# input_resolution=(13, 128, 256), L=13*128*256=425984
layer = FuserLayer(
    dim=192,
    input_resolution=(13, 128, 256),
    depth=2,
    num_heads=6,
    window_size=(2, 6, 12)
)
x = torch.randn(4, 425984, 192)
out = layer(x)
print('output:  ', out.shape)
print(f'expected: torch.Size([4, 425984, 192])')