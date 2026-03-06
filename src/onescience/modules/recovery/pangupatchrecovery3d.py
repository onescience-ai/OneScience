import torch
from torch import nn

class PanguPatchRecovery3D(nn.Module):

    """
        Pangu-Weather 模型中的三维 Patch 恢复模块，用三维反卷积将 Patch 特征还原为原始层数与空间分辨率的三维场，并裁剪掉补零边界。

        Args:
            img_size (tuple[int, int, int]): 输出目标场的尺寸 (L, H, W)，分别对应垂直层数和空间网格
            patch_size (tuple[int, int, int]): 三维 Patch 大小 (patch_l, patch_h, patch_w)，即反卷积的 kernel_size 与 stride
            in_chans (int): 输入特征通道数
            out_chans (int): 输出场通道数

        形状:
            输入:  x 形状为 (B, in_chans, L', H', W')
            输出:  y 形状为 (B, out_chans, img_size[0], img_size[1], img_size[2])

        Example:
            >>> recovery3d = PanguPatchRecovery3D(
            ...     img_size=(13, 721, 1440),
            ...     patch_size=(2, 4, 4),
            ...     in_chans=384,
            ...     out_chans=5,
            ... )
            >>> x = torch.randn(2, 384, 7, 181, 360)
            >>> y = recovery3d(x)
            >>> y.shape
            torch.Size([2, 5, 13, 721, 1440])
    """
    def __init__(self, img_size = (13, 721, 1440), 
                 patch_size = (2, 4, 4),
                 in_chans = 192*2, 
                 out_chans = 5):
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
    


      