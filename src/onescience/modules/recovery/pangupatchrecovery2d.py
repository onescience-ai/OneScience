import torch
from torch import nn


class PatchRecovery2D(nn.Module):
    """
    改编自 WeatherLearn 项目 https://github.com/lizhuoq/WeatherLearn
    Patch 嵌入恢复为二维图像

    参数:
        img_size (tuple[int]): 图像的空间尺寸，格式为 (Lat, Lon)，即纬度和经度方向的大小
        patch_size (tuple[int]): 每个 patch 的大小，格式为 (Lat, Lon)
        in_chans (int): 输入特征的通道数
        out_chans (int): 输出图像的通道数
    """

    def __init__(self, 
                  img_size = (721, 1440),
                  patch_size = (4, 4),
                  in_chans = 192*2,
                  out_chans = 4):
        super().__init__()
        self.img_size = img_size
        self.conv = nn.ConvTranspose2d(in_chans, out_chans, patch_size, patch_size)

    def forward(self, x):
        output = self.conv(x)
        _, _, H, W = output.shape
        h_pad = H - self.img_size[0]
        w_pad = W - self.img_size[1]

        padding_top = h_pad // 2
        padding_bottom = int(h_pad - padding_top)

        padding_left = w_pad // 2
        padding_right = int(w_pad - padding_left)

        return output[
            :, :, padding_top : H - padding_bottom, padding_left : W - padding_right
        ]