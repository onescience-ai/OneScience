import torch
from onescience.modules.attention.EarthAttention import EarthAttention3D


attn = EarthAttention3D(
    dim=192,
    input_resolution=(13, 128, 256), 
    window_size=(1, 8, 8),
    num_heads=6
)
# num_pl=pressure_levels // window_size_1 = 13//1=13
# num_lat=lat // window_size_2 = 128//8=16
# num_lon=lon // window_size_3 = 256//8=32
# 输入: (B*num_lon, num_pl*num_lat, Wpl*Wlat*Wlon, C)
#      = (4*32, 13*16, 1*8*8, 192)
#      = (128, 208, 64, 192)
x = torch.randn(128, 208, 64, 192)
out = attn(x)
print('output:  ', out.shape)
print(f'expected: torch.Size([128, 208, 64, 192])')