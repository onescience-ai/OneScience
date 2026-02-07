import torch
from onescience.modules.attention.EarthAttention import OneEarthAttention


# 2D地表变量注意力
attn = OneEarthAttention(
    dim=128,
    input_resolution=(128, 256),
    window_size=(8, 8),
    num_heads=4,
    style='pangu'
)
# num_lat = 128 // 8 = 16
# num_lon = 256 // 8 = 32
# 输入: (B*num_lon, num_lat, Wlat*Wlon, C)
#      = (4*32, 16, 8*8, 128)
#      = (128, 16, 64, 128)
x = torch.randn(128, 16, 64, 128)
out = attn(x)
print('Function: 2D OneEarthAttention')
print(f'output shape: {out.shape}')
print( 'target shape: torch.Size([128, 16, 64, 128])\n')

# 3D大气变量注意力
attn = OneEarthAttention(
    dim=192,
    input_resolution=(13, 128, 256),
    window_size=(1, 8, 8),
    num_heads=6,
    style='pangu'
)
# num_pl = 13 // 1 = 13
# num_lat = 128 // 8 = 16
# num_lon = 256 // 8 = 32
# 输入: (B*num_lon, num_pl*num_lat, Wpl*Wlat*Wlon, C)
#      = (4*32, 13*16, 1*8*8, 192)
#      = (128, 208, 64, 192)
x = torch.randn(128, 208, 64, 192)
out = attn(x)
print('Function: 3D OneEarthAttention')
print(f'output shape: {out.shape}')
print( 'target shape: torch.Size([128, 208, 64, 192])\n')
