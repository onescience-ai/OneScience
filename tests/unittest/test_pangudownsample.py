import torch
from onescience.modules import OneSample
import warnings

# 忽略有关 'torch.meshgrid' 的警告
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")

Batch = 2
input_resolution = (181, 360)
output_resolution = (91, 180)
in_dim = 192

# Pangu-Weather surface 分支
surface_downsample = OneSample(
    style="PanguDownSample",
    input_resolution=input_resolution,
    output_resolution=output_resolution,
    in_dim=in_dim
).cuda()

surface_x = torch.randn(Batch, input_resolution[0] * input_resolution[1], in_dim).cuda()

surface_out = surface_downsample(surface_x)

surface_target_shape = torch.Size(
    [Batch, output_resolution[0] * output_resolution[1], 2 * in_dim]
)

print('Function: Pangu Down Sample Surface Forward')
print(f'output shape: {surface_out.shape}')
print(f'target shape: {surface_target_shape}')

if surface_out.shape == surface_target_shape:
    print('Unit test Pass\n')
else:
    print('Unit test not pass\n')


input_resolution = (8, 181, 360)
output_resolution = (8, 91, 180)
in_dim = 192

# Pangu-Weather upper-air 分支
upper_air_downsample = OneSample(
    style="PanguDownSample",
    input_resolution=input_resolution,
    output_resolution=output_resolution,
    in_dim=in_dim
).cuda()

upper_air_x = torch.randn(
    Batch,
    input_resolution[0] * input_resolution[1] * input_resolution[2],
    in_dim
).cuda()

upper_air_out = upper_air_downsample(upper_air_x)

upper_air_target_shape = torch.Size(
    [Batch, output_resolution[0] * output_resolution[1] * output_resolution[2], 2 * in_dim]
)

print('Function: Pangu Down Sample Upper Air Forward')
print(f'output shape: {upper_air_out.shape}')
print(f'target shape: {upper_air_target_shape}')

if upper_air_out.shape == upper_air_target_shape:
    print('Unit test Pass\n')
else:
    print('Unit test not pass\n')
