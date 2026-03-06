import torch
from onescience.models.fuxi import Fuxi
import warnings

# 忽略有关 'torch.meshgrid' 的警告
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")

model = Fuxi(
    img_size=(2, 721, 1440), 
    patch_size=(2, 4, 4), 
    in_chans=70, 
    out_chans=70,
    embed_dim=1536, 
    num_groups=32, 
    num_heads=8, 
    window_size=7)
x = torch.randn(2, 2, 70, 721, 1440)
x = x.permute(0, 2, 1, 3, 4) 
out = model(x)
print('Function: FuXi Model Forward')
print(f'output shape: {out.shape}')
print( 'target shape: torch.Size([2, 70, 721, 1440])\n')