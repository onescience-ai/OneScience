import torch
from torch import nn


class PatchEmbed2D(nn.Module):
    """
    改编自 WeatherLearn 项目 https://github.com/lizhuoq/WeatherLearn
    二维图像到 Patch 嵌入.

    Args:
        img_size (tuple[int]): 输入图像的尺寸.
        patch_size (tuple[int]): 每个 Patch 的大小.
        in_chans (int): 输入图像的通道数.
        embed_dim(int): 每个 Patch 嵌入后的向量维度.
        norm_layer (nn.Module, optional): 对 Patch 嵌入结果进行归一化的层，默认值为 None。常见选项：nn.LayerNorm 或 nn.BatchNorm2d.
    """

    def __init__(self, img_size, patch_size, in_chans, embed_dim, norm_layer=None):
        super().__init__()
        self.img_size = img_size
        height, width = img_size
        h_patch_size, w_path_size = patch_size
        padding_left = padding_right = padding_top = padding_bottom = 0

        h_remainder = height % h_patch_size
        w_remainder = width % w_path_size

        if h_remainder:
            h_pad = h_patch_size - h_remainder
            padding_top = h_pad // 2
            padding_bottom = int(h_pad - padding_top)

        if w_remainder:
            w_pad = w_path_size - w_remainder
            padding_left = w_pad // 2
            padding_right = int(w_pad - padding_left)

        self.pad = nn.ZeroPad2d(
            (padding_left, padding_right, padding_top, padding_bottom)
        )
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        x = self.pad(x)
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x


class PatchEmbed3D(nn.Module):
    """
    改编自 WeatherLearn 项目 https://github.com/lizhuoq/WeatherLearn
    三维图像到 Patch 嵌入

    Args:
        img_size (tuple[int]): 输入图像的尺寸.
        patch_size (tuple[int]): 每个 Patch 的大小.
        in_chans (int): 输入图像的通道数.
        embed_dim(int): 每个 Patch 嵌入后的向量维度.
        norm_layer (nn.Module, optional): 对 Patch 嵌入结果进行归一化的层，默认值为 None.
    """

    def __init__(self, img_size, patch_size, in_chans, embed_dim, norm_layer=None):
        super().__init__()
        self.img_size = img_size
        level, height, width = img_size
        l_patch_size, h_patch_size, w_patch_size = patch_size
        padding_left = (
            padding_right
        ) = padding_top = padding_bottom = padding_front = padding_back = 0

        l_remainder = level % l_patch_size
        h_remainder = height % l_patch_size
        w_remainder = width % w_patch_size

        if l_remainder:
            l_pad = l_patch_size - l_remainder
            padding_front = l_pad // 2
            padding_back = l_pad - padding_front
        if h_remainder:
            h_pad = h_patch_size - h_remainder
            padding_top = h_pad // 2
            padding_bottom = h_pad - padding_top
        if w_remainder:
            w_pad = w_patch_size - w_remainder
            padding_left = w_pad // 2
            padding_right = w_pad - padding_left

        self.pad = nn.ZeroPad3d(
            (
                padding_left,
                padding_right,
                padding_top,
                padding_bottom,
                padding_front,
                padding_back,
            )
        )
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor):
        B, C, L, H, W = x.shape
        x = self.pad(x)
        x = self.proj(x)
        if self.norm:
            x = self.norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        return x


class PatchRecovery2D(nn.Module):
    """
    改编自 WeatherLearn 项目 https://github.com/lizhuoq/WeatherLearn
    Patch 嵌入恢复为二维图像.

    参数:
        img_size (tuple[int]): 图像的空间尺寸，格式为 (Lat, Lon)，即纬度和经度方向的大小
        patch_size (tuple[int]): 每个 patch 的大小，格式为 (Lat, Lon)
        in_chans (int): 输入特征的通道数
        out_chans (int): 输出图像的通道数
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
