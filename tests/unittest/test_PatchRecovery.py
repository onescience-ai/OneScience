import torch
from onescience.modules.patch.PatchRecovery import OnePatchRecovery
import warnings

# 忽略有关 'torch.meshgrid' 的警告
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")
# 2D图像恢复
patch_recovery = OnePatchRecovery(
    img_size=(128, 256),
    patch_size=(4, 4),
    in_chans=96,
    out_chans=3,
    style='pangu'
)
x = torch.randn(8, 96, 32, 64)
out = patch_recovery(x)
print('Function: 2D PatchRecovery')
print(f'output shape: {out.shape}')
print( 'target shape: torch.Size([8, 3, 128, 256])\n')
# 3D图像恢复
patch_recovery = OnePatchRecovery(
    img_size=(13, 128, 256),
    patch_size=(1, 4, 4),
    in_chans=192,
    out_chans=5,
    style='pangu'
)
x = torch.randn(4, 192, 13, 32, 64)
out = patch_recovery(x)
print('Function: 2D PatchRecovery')
print(f'output shape: {out.shape}')
print( 'target shape: torch.Size([4, 5, 13, 128, 256])\n')

