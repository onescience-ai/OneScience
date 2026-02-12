import torch
from torch import nn

class PanguEmbedding3D(nn.Module):
    """
        将三维图像分割为不重叠的 patch 并嵌入到向量空间。

        Args:
            img_size (tuple[int, int, int]): 输入图像尺寸 (P, H, W)
            patch_size (tuple[int, int, int]): 每个 patch 的大小 (patch_p, patch_h, patch_w)
            in_chans (int): 输入图像通道数
            embed_dim (int): 每个 patch 嵌入后的向量维度
            norm_layer (nn.Module, optional): 归一化层，默认为 None

        形状:
            输入: (B, C, P, H, W)
            输出: (B, embed_dim, P', H', W'), 其中 P' = ⌈P / patch_p⌉, H' = ⌈H / patch_h⌉, W' = ⌈W / patch_w⌉

        Example:
            >>> patch_embed = PatchEmbed3D(
            ...     img_size=(13, 128, 256),
            ...     patch_size=(1, 4, 4),
            ...     in_chans=5,
            ...     embed_dim=192
            ... )
            >>> x = torch.randn(4, 5, 13, 128, 256)
            >>> out = patch_embed(x)
            >>> out.shape
            torch.Size([4, 192, 13, 32, 64])

    """

    def __init__(self, *args, **kwargs):

        super().__init__()

        self.img_size = (13, 721, 1440)
        level, height, width = self.img_size
        l_patch_size, h_patch_size, w_patch_size = (2, 4, 4)
        in_chans = 5
        embed_dim = 192
        stride = patch_size = (2, 4, 4)
        norm_layer = None

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
            in_chans, embed_dim, kernel_size=patch_size, stride=stride
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