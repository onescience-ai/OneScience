import torch
from torch import nn


class PatchRecovery2D(nn.Module):
    """
        将 patch 嵌入恢复为二维图像。

        Args:
            img_size (tuple[int, int]): 目标图像尺寸 (H, W)
            patch_size (tuple[int, int]): 每个 patch 的大小 (patch_h, patch_w)
            in_chans (int): 输入特征通道数
            out_chans (int): 输出图像通道数

        形状:
            输入: (B, in_chans, H', W')
            输出: (B, out_chans, H, W)

        Example:
            >>> patch_recovery = PatchRecovery2D(
            ...     img_size=(128, 256),
            ...     patch_size=(4, 4),
            ...     in_chans=96,
            ...     out_chans=3
            ... )
            >>> x = torch.randn(8, 96, 32, 64)
            >>> out = patch_recovery(x)
            >>> out.shape
            torch.Size([8, 3, 128, 256])

    """

    def __init__(self, img_size, patch_size, in_chans, out_chans):
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


class PatchRecovery3D(nn.Module):
    """
        将 patch 嵌入恢复为二维图像。

        Args:
            img_size (tuple[int, int, int]): 目标图像尺寸 (P, H, W)
            patch_size (tuple[int, int, int]): 每个 patch 的大小 (patch_p, patch_h, patch_w)
            in_chans (int): 输入特征通道数
            out_chans (int): 输出图像通道数

        形状:
            输入: (B, in_chans, P', H', W')
            输出: (B, out_chans, P, H, W)

        Example:
            >>> patch_recovery = PatchRecovery3D(
            ...     img_size=(13, 128, 256),
            ...     patch_size=(1, 4, 4),
            ...     in_chans=192,
            ...     out_chans=5
            ... )
            >>> x = torch.randn(4, 192, 13, 32, 64)
            >>> out = patch_recovery(x)
            >>> out.shape
            torch.Size([4, 5, 13, 128, 256])

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
