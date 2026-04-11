import torch
from onescience.models.fuxi import Fuxi
import warnings
from onescience.memory.checkpoint import replace_function
# 忽略有关 'torch.meshgrid' 的警告
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Fuxi(
    img_size=(2, 721, 1440), 
    patch_size=(2, 4, 4), 
    in_chans=70, 
    out_chans=70,
    embed_dim=1536, 
    num_groups=32, 
    num_heads=8, 
    window_size=7).to(device)
x = torch.randn(2, 2, 70, 721, 1440).to(device)
x = x.permute(0, 2, 1, 3, 4) 
with replace_function(model, ["cube_embedding", "u_transformer"], False):
    out = model(x)
print('Function: FuXi Model Forward')
print(f'output shape: {out.shape}')
print( 'target shape: torch.Size([2, 70, 721, 1440])\n')