import torch
from torch import nn
from torch.nn import functional as F
from timm.layers.helpers import to_2tuple
from timm.models.swin_transformer_v2 import SwinTransformerV2Stage

from typing import Sequence
from onescience.modules.func_utils.fuxi_utils import get_pad2d


class FuxiUpSample(nn.Module):
    """
        FuXi 模型的二维上采样模块，先用步长为 2 的反卷积将空间分辨率放大一倍，再通过若干残差卷积块细化特征。

        Args:
            in_chans (int): 输入特征通道数
            out_chans (int): 输出特征通道数
            num_groups (int): GroupNorm 的分组数
            num_residuals (int): 残差卷积块的数量（每个块包含 Conv2d + GroupNorm + SiLU）

        形状:
            输入:  x 形状为 (B, in_chans, H, W)
            输出:  y 形状为 (B, out_chans, H_out, W_out)，其中 H_out = H * 2，W_out = W * 2

        Example:
            >>> up = FuxiUpSample(in_chans=3072, out_chans=1536, num_groups=32, num_residuals=2)
            >>> x = torch.randn(2, 3072, 90, 180)
            >>> y = up(x)
            >>> y.shape
            torch.Size([2, 1536, 180, 360])
    """
    def __init__(self, 
                 in_chans=1536*2, 
                 out_chans=1536, 
                 num_groups=32, 
                 num_residuals=2):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2)

        blk = []
        for i in range(num_residuals):
            blk.append(nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1))
            blk.append(nn.GroupNorm(num_groups, out_chans))
            blk.append(nn.SiLU())

        self.b = nn.Sequential(*blk)

    def forward(self, x):
        x = self.conv(x)

        shortcut = x

        x = self.b(x)

        return x + shortcut

