import torch
from onescience.modules import OneRecovery
import warnings


warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")

Batch = 2
img_size = (721, 1440)
patch_size = (4, 4)
in_chans = 384
out_chans = 7

surface_recovery = OneRecovery(
    style="PanguPatchRecovery",
    img_size=img_size,
    patch_size=patch_size,
    in_chans=in_chans,
    out_chans=out_chans,
)

surface_x = torch.randn(Batch, in_chans, 181, 360)
surface_out = surface_recovery(surface_x)
surface_target_shape = torch.Size([Batch, out_chans, *img_size])

print("Function: Pangu Patch Recovery Surface Forward")
print(f"output shape: {surface_out.shape}")
print(f"target shape: {surface_target_shape}")

if surface_out.shape == surface_target_shape:
    print("Unit test Pass\n")
else:
    print("Unit test not pass\n")


img_size = (13, 721, 1440)
patch_size = (2, 4, 4)
out_chans = 5

upper_air_recovery = OneRecovery(
    style="PanguPatchRecovery",
    img_size=img_size,
    patch_size=patch_size,
    in_chans=in_chans,
    out_chans=out_chans,
)

upper_air_x = torch.randn(Batch, in_chans, 7, 181, 360)
upper_air_out = upper_air_recovery(upper_air_x)
upper_air_target_shape = torch.Size([Batch, out_chans, *img_size])

print("Function: Pangu Patch Recovery Upper Air Forward")
print(f"output shape: {upper_air_out.shape}")
print(f"target shape: {upper_air_target_shape}")

if upper_air_out.shape == upper_air_target_shape:
    print("Unit test Pass\n")
else:
    print("Unit test not pass\n")
