import torch
from onescience.models.pangu import Pangu
import warnings

# 忽略有关 'torch.meshgrid' 的警告
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")

model = Pangu(img_size=(721, 1440),
              patch_size=[2, 4, 4],
              embed_dim=192,
              num_heads=[6, 12, 12, 6],
              window_size=[2, 6, 12],
              )
x = torch.randn(2, 69, 721, 1440)
surface_mask = torch.randn(2, 3, 721, 1440)

invar_surface = x[:, :4, :, :]
invar_upper_air = x[:, 4:, :, :]
invar = torch.concat([invar_surface, surface_mask, invar_upper_air], dim=1)

out_surface, out_upper_air = model(invar)
out_upper_air = out_upper_air.reshape(invar_upper_air.shape)
print('Function: Pangu Model Forward Pass')
print(f'output shape: {out_surface.shape}, {out_upper_air.shape}')
print( 'target shape: torch.Size([2, 70, 721, 1440])\n')cd 