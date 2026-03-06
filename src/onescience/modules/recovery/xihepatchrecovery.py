import torch
from torch import nn


class XihePatchRecovery(nn.Module):

    """
        Pangu-Weather 模型中的二维 Patch 恢复模块，用反卷积将 Patch 特征还原为原始空间分辨率的二维场，并裁剪掉补零边界。

        Args:
            img_size (tuple[int, int]): 输出目标图像尺寸 (H, W)
            patch_size (tuple[int, int]): Patch 大小 (patch_h, patch_w)，即反卷积的 kernel_size 与 stride
            in_chans (int): 输入特征通道数
            out_chans (int): 输出图像通道数

        形状:
            输入:  x 形状为 (B, in_chans, H', W')
            输出:  y 形状为 (B, out_chans, img_size[0], img_size[1])

        Example:
            >>> recovery = PanguPatchRecovery2D(
            ...     img_size=(721, 1440),
            ...     patch_size=(4, 4),
            ...     in_chans=384,
            ...     out_chans=4,
            ... )
            >>> x = torch.randn(2, 384, 181, 360)
            >>> y = recovery(x)
            >>> y.shape
            torch.Size([2, 4, 721, 1440])
    """
    def __init__(self, 
                img_size = (2041, 4320),
                patch_size = (6, 12),
                in_chans = 192,
                out_chans = 96):
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