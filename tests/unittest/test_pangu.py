import torch
from onescience.models.pangu import Pangu
import warnings
from onescience.memory.checkpoint import replace_function

# 忽略有关 'torch.meshgrid' 的警告
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Pangu(img_size=(721, 1440),
              patch_size=[2, 4, 4],
              embed_dim=192,
              num_heads=[6, 12, 12, 6],
              window_size=[2, 6, 12],
              ).to(device)
x = torch.randn(2, 69, 721, 1440).to(device)
surface_mask = torch.randn(2, 3, 721, 1440).to(device)

invar_surface = x[:, :4, :, :]
invar_upper_air = x[:, 4:, :, :]
invar = torch.concat([invar_surface, surface_mask, invar_upper_air], dim=1)

with replace_function(model, ["layer1", "layer2", "layer3", "layer4"], False):
    out_surface, out_upper_air = model(invar)
out_upper_air = out_upper_air.reshape(invar_upper_air.shape)
print('Function: Pangu Model Forward')
print(f'output shape: {out_surface.shape}, {out_upper_air.shape}')
print( 'target shape: torch.Size([2, 4, 721, 1440]), torch.Size([2, 65, 721, 1440])\n')