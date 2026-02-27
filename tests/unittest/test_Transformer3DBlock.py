import torch
from onescience.modules.block.Transformer3DBlock import OneTransformer3DBlock
import warnings

# 忽略有关 'torch.meshgrid' 的警告
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")

# 基础用法
# input_resolution=(13, 128, 256), L=13*128*256=425984
block = OneTransformer3DBlock(
    dim=192,
    input_resolution=(13, 128, 256),
    num_heads=6,
    window_size=(2, 6, 12),
    shift_size=(1, 3, 6),
    style='pangu'
)
x = torch.randn(4, 425984, 192)
out = block(x)
print('Function: basic OneTransformer3DBlock')
print(f'output shape: {out.shape}')
print( 'target shape: torch.Size([4, 425984, 192])\n')

# 使用默认window_size和shift_size
# 偶数层使用shift_size=(0,0,0)，奇数层使用默认shift
blocks = torch.nn.ModuleList([
    OneTransformer3DBlock(
        dim=192,
        input_resolution=(13, 128, 256),
        num_heads=6,
        window_size=(2, 6, 12),
        shift_size=(0, 0, 0) if i % 2 == 0 else None,
        style='pangu'
    )
    for i in range(2)
])
print('Function: multi-depth OneTransformer3DBlock')
print(f'output shape: {out.shape}')
print( 'target shape: torch.Size([4, 425984, 192])\n')