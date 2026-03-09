import torch
from torch import nn
from torch.nn import functional as F
from timm.layers.helpers import to_2tuple
from timm.models.swin_transformer_v2 import SwinTransformerV2Stage

from typing import Sequence
from onescience.modules.func_utils.fuxi_utils import get_pad2d


class FuxiDownSample(nn.Module):
    """
        FuXi 模型的二维下采样模块，先用步长为 2 的卷积将空间分辨率减半，再通过若干残差卷积块进行特征变换。

        Args:
            in_chans (int): 输入特征通道数
            out_chans (int): 输出特征通道数
            num_groups (int): GroupNorm 的分组数
            num_residuals (int): 残差卷积块的数量（每个块包含 Conv2d + GroupNorm + SiLU）

        形状:
            输入:  x 形状为 (B, in_chans, H, W)
            输出:  y 形状为 (B, out_chans, H_out, W_out)，其中 H_out = H // 2，W_out = W // 2

        Example:
            >>> down = FuxiDownSample(in_chans=1536, out_chans=1536, num_groups=32, num_residuals=2)
            >>> x = torch.randn(2, 1536, 180, 360)
            >>> y = down(x)
            >>> y.shape
            torch.Size([2, 1536, 90, 180])
    """
    def __init__(self, 
                 in_chans=1536, 
                 out_chans=1536, 
                 num_groups=32, 
                 num_residuals=2):
        super().__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=(3, 3), stride=2, padding=1)

        blk = []
        for i in range(num_residuals):
            blk.append(nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1))
            blk.append(nn.GroupNorm(num_groups, out_chans))
            blk.append(nn.SiLU())

        self.b = nn.Sequential(*blk)

    def forward(self, x):
        _, _, h, w = x.shape
        x = self.conv(x)

        shortcut = x

        x = self.b(x)

        res = x + shortcut
        if h % 2 != 0:
            res = res[:, :, :-1, :]
        if w % 2 != 0:
            res = res[:, :, :, :-1]
        return res
