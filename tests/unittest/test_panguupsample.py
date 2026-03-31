import torch
from onescience.modules import OneSample
import warnings


warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")

Batch = 2
input_resolution = (91, 180)
output_resolution = (181, 360)
in_dim = 384
out_dim = 192

surface_upsample = OneSample(
    style="PanguUpSample",
    input_resolution=input_resolution,
    output_resolution=output_resolution,
    in_dim=in_dim,
    out_dim=out_dim,
).cuda()

surface_x = torch.randn(Batch, input_resolution[0] * input_resolution[1], in_dim).cuda()
surface_out = surface_upsample(surface_x)
surface_target_shape = torch.Size(
    [Batch, output_resolution[0] * output_resolution[1], out_dim]
)

print("Function: Pangu Up Sample Surface Forward")
print(f"output shape: {surface_out.shape}")
print(f"target shape: {surface_target_shape}")

if surface_out.shape == surface_target_shape:
    print("Unit test Pass\n")
else:
    print("Unit test not pass\n")


input_resolution = (8, 91, 180)
output_resolution = (8, 181, 360)

upper_air_upsample = OneSample(
    style="PanguUpSample",
    input_resolution=input_resolution,
    output_resolution=output_resolution,
    in_dim=in_dim,
    out_dim=out_dim,
).cuda()

upper_air_x = torch.randn(
    Batch,
    input_resolution[0] * input_resolution[1] * input_resolution[2],
    in_dim,
).cuda()
upper_air_out = upper_air_upsample(upper_air_x)
upper_air_target_shape = torch.Size(
    [Batch, output_resolution[0] * output_resolution[1] * output_resolution[2], out_dim]
)

print("Function: Pangu Up Sample Upper Air Forward")
print(f"output shape: {upper_air_out.shape}")
print(f"target shape: {upper_air_target_shape}")

if upper_air_out.shape == upper_air_target_shape:
    print("Unit test Pass\n")
else:
    print("Unit test not pass\n")
