import torch
from torch import nn
from torch.nn import functional as F
from timm.layers.helpers import to_2tuple
from timm.models.swin_transformer_v2 import SwinTransformerV2Stage

from typing import Sequence
from onescience.modules.func_utils.fuxi_utils import get_pad2d
from onescience.modules.sample.onesample import OneSample


class FuxiTransformer(nn.Module):
    """
        FuXi 模型的核心 Transformer 处理模块。

        采用"下采样 → Swin Transformer V2 → 上采样"的 U 形结构，在低分辨率特征图上
        进行深层注意力计算以降低计算量，并通过跳跃连接将下采样前的特征与注意力输出拼接
        后恢复原始分辨率。使用 SwinTransformerV2Stage 作为主干，支持 ZeroPad + Crop
        处理分辨率与窗口大小不整除的情况。

        Args:
            embed_dim (int, optional): 输入特征的通道数，默认为 1536。
            num_groups (int 或 tuple[int, int], optional): GroupNorm 的分组数，
                传入单个 int 时自动扩展为 2 元组，默认为 32。
            input_resolution (tuple[int, int], optional): 下采样后的特征图分辨率
                (lat, lon)，即 SwinTransformerV2Stage 的输入分辨率，
                默认为 (90, 180)。
            num_heads (int, optional): Swin Transformer 的注意力头数，默认为 8。
            window_size (int 或 tuple[int, int], optional): 窗口注意力的窗口大小，
                传入单个 int 时自动扩展为 2 元组，默认为 7。
            depth (int, optional): SwinTransformerV2Stage 的 Block 层数，默认为 48。

        形状:
            - 输入 x: (B, embed_dim, lat, lon)
                其中 lat, lon 为下采样前的原始分辨率（约为 input_resolution 的 2 倍）
            - 输出:   (B, embed_dim, lat, lon)，分辨率与通道数均不变

        Examples:
            >>> # 典型 FuXi 配置
            >>> # 原始输入分辨率: (180, 360)，下采样后: (90, 180)
            >>> # depth=48 对应 FuXi 论文中的深层 Swin Transformer 堆叠
            >>> transformer = FuxiTransformer(
            ...     embed_dim=1536,
            ...     num_groups=32,
            ...     input_resolution=(90, 180),
            ...     num_heads=8,
            ...     window_size=7,
            ...     depth=48,
            ... )
            >>> x = torch.randn(2, 1536, 180, 360)  # (B, C, lat, lon)
            >>> out = transformer(x)
            >>> out.shape
            torch.Size([2, 1536, 180, 360])
    """

    def __init__(self, 
                 embed_dim=1536,
                 num_groups=32, 
                 input_resolution=(90, 180),
                 num_heads=8, 
                 window_size=7, 
                 depth=48):
        super().__init__()
        
        num_groups = to_2tuple(num_groups)
        window_size = to_2tuple(window_size)
        padding = get_pad2d(input_resolution, window_size)
        padding_left, padding_right, padding_top, padding_bottom = padding
        self.padding = padding
        self.pad = nn.ZeroPad2d(padding)
        input_resolution = list(input_resolution)
        input_resolution[0] = input_resolution[0] + padding_top + padding_bottom
        input_resolution[1] = input_resolution[1] + padding_left + padding_right
        self.down = OneSample(style="FuxiDownSample", in_chans=embed_dim, out_chans=embed_dim, num_groups=num_groups[0])
        self.layer = SwinTransformerV2Stage(embed_dim, embed_dim, input_resolution, depth, num_heads, window_size)
        self.up = OneSample(style="FuxiUpSample", in_chans=embed_dim*2, out_chans=embed_dim, num_groups=num_groups[0])

    def forward(self, x):
        B, C, Lat, Lon = x.shape
        padding_left, padding_right, padding_top, padding_bottom = self.padding
        x = self.down(x)

        shortcut = x

        # pad
        x = self.pad(x)
        _, _, pad_lat, pad_lon = x.shape

        x = x.permute(0, 2, 3, 1)  # B Lat Lon C
        x = self.layer(x)
        x = x.permute(0, 3, 1, 2)

        # crop
        x = x[:, :, padding_top: pad_lat - padding_bottom, padding_left: pad_lon - padding_right]

        # concat
        x = torch.cat([shortcut, x], dim=1)  # B 2*C Lat Lon

        x = self.up(x)
        return x