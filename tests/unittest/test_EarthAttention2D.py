import torch
from onescience.modules.attention.EarthAttention import EarthAttention2D


attn = EarthAttention2D(
    dim=128,
    input_resolution=(128, 256),
    window_size=(8, 8),
    num_heads=4
)
# num_lat=lat // window_size_2 = 128//8=16
# num_lon=lon // window_size_3 = 256//8=32
# 输入: (B*num_lon, num_lat, Wlat*Wlon, C)
#      = (4*32, 16, 8*8, 128)
#      = (128, 16, 64, 128)
x = torch.randn(128, 16, 64, 128)
out = attn(x)
print('output:  ', out.shape)
print(f'expected: torch.Size([128, 16, 64, 128])')