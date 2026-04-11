import torch
from onescience.models.oceancast import OceanCast
import warnings

# 忽略有关 'torch.meshgrid' 的警告
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = OceanCast(img_size=(160, 360), patch_size=(4, 4), in_chans=72, out_chans=24).to(device)
x = torch.randn(2, 72, 160, 360).to(device)

out = model(x)
print('Function: OceanCast Model Forward')
print(f'output shape: {out.shape}')
print( 'target shape: torch.Size([2, 24, 160, 360])\n')