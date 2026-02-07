import torch
from torch import nn

class PanGuPatchRecovery3D(nn.Module):
    """
    改编自 WeatherLearn 项目 https://github.com/lizhuoq/WeatherLearn
    Patch 嵌入恢复为三维图像.

    参数:
        img_size (tuple[int]): 图像的空间尺寸，格式为 (Lat, Lon)，即纬度和经度方向的大小
        patch_size (tuple[int]): 每个 patch 的大小，格式为 (Lat, Lon)
        in_chans (int): 输入特征的通道数
        out_chans (int): 输出图像的通道数
    """

    def __init__(self, img_size, patch_size, in_chans, out_chans):
        super().__init__()
        self.img_size = img_size
        self.conv = nn.ConvTranspose3d(in_chans, out_chans, patch_size, patch_size)

    def forward(self, x: torch.Tensor):
        output = self.conv(x)
        _, _, Pl, Lat, Lon = output.shape

        pl_pad = Pl - self.img_size[0]
        lat_pad = Lat - self.img_size[1]
        lon_pad = Lon - self.img_size[2]

        padding_front = pl_pad // 2
        padding_back = pl_pad - padding_front

        padding_top = lat_pad // 2
        padding_bottom = lat_pad - padding_top

        padding_left = lon_pad // 2
        padding_right = lon_pad - padding_left

        return output[
            :,
            :,
            padding_front : Pl - padding_back,
            padding_top : Lat - padding_bottom,
            padding_left : Lon - padding_right,
        ]
    

class OnePatch(nn.module):
    def __init__(self, style="PanGuPatchRecovery3D"):
        
        if self.style == "PanGuPatchRecovery3D":
            self.PanGuPatchRecovery3D = PanGuPatchRecovery3D(self, dim, input_resolution, window_size, num_heads,)
        else:
            raise NotImplementedError
      