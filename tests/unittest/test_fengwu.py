import torch
from onescience.models.fengwu import Fengwu
import warnings
from onescience.memory.checkpoint import replace_function
# 忽略有关 'torch.meshgrid' 的警告
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Fengwu(img_size=[721, 1440],
                   pressure_level=37,
                   embed_dim=192,
                   patch_size=[4, 4],
                   num_heads=[6, 12, 12, 6],
                   window_size=[2, 6, 12],
                   ).to(device)
x = torch.randn(2, 189, 721, 1440).to(device)
surface = x[:, :4, :, :]
z = x[:, 4:41, :, :]
r = x[:, 41:78, :, :]
u = x[:, 78:115, :, :]
v = x[:, 115:152, :, :]
t = x[:, 152:189, :, :]
with replace_function(model, 
                      ["encoder_surface","encoder_z","encoder_r","encoder_u","encoder_v","encoder_t","fuser"],
                      False):
    surface_p, z_p, r_p, u_p, v_p, t_p = model(surface, z, r, u, v, t)
out = torch.concat([surface_p, z_p, r_p, u_p, v_p, t_p],dim=1)
print('Function: Fengwu Model Forward')
print(f'output shape: {out.shape}')
print( 'target shape: torch.Size([2, 189, 721, 1440])\n')