import torch
from onescience.models.fourcastnet import FourCastNet
import warnings

# 忽略有关 'torch.meshgrid' 的警告
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = FourCastNet().to(device)
x = torch.randn(2, 19, 720, 1440).to(device)

out = model(x)
print('Function: FourCastNet Model Forward')
print(f'output shape: {out.shape}')
print( 'target shape: torch.Size([2, 19, 720, 1440])\n')