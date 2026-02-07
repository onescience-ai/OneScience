import torch
from onescience.models.pangu import Pangu
import warnings

# 忽略有关 'torch.meshgrid' 的警告
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")

model = Pangu(
    img_size=(721, 1440),
    patch_size=(2, 4, 4),
    embed_dim=192,
    num_heads=(6, 12, 12, 6),
    window_size=(2, 6, 12)
)
x = torch.randn(1, 69+3, 721, 1440)
surface_out, upper_air_out = model(x)
print('Function: Pangu Model Forward Pass')
print(f'output shape: {surface_out.shape.shape}, {upper_air_out.shape}')
print( 'target shape: (torch.Size([2, 4, 721, 1440]), torch.Size([2, 5, 13, 721, 1440]))\n')