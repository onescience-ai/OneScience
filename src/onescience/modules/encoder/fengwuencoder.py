from collections.abc import Sequence

import torch
from torch import nn

from onescience.modules.embedding.oneembedding import OneEmbedding
from onescience.modules.sample.onesample import OneSample
from onescience.modules.transformer.onetransformer import OneTransformer

class FengWuEncoder(nn.Module):
    """
        FengWu 模型的二维层次化编码器，对高分辨率气象场进行编码并输出中分辨率特征与高分辨率 skip 连接。

        Args:
            input_resolution (tuple[int, int]): 高分辨率编码阶段的空间分辨率 (H1, W1)
            middle_resolution (tuple[int, int]): 下采样后中分辨率编码阶段的空间分辨率 (Hm, Wm)
            in_chans (int): 输入气象变量通道数
            img_size (tuple[int, int]): 原始输入场尺寸 (H, W)
            patch_size (tuple[int, int]): Patch 大小 (patch_h, patch_w)
            dim (int): 高分辨率阶段的特征维度
            depth (int): 高分辨率 Transformer 块层数
            depth_middle (int): 中分辨率 Transformer 块层数
            num_heads (int | tuple[int, int]): 多头自注意力头数配置（单个或 (high, middle)）
            window_size (int | tuple[int, int]): 窗口注意力窗口大小
            mlp_ratio (float): 前馈网络隐藏层与特征维度的比例
            qkv_bias (bool): 是否在 QKV 投影中使用偏置
            qk_scale (float | None): QK 点积缩放因子
            drop (float): 特征上的 dropout 比例
            attn_drop (float): 注意力权重上的 dropout 比例
            drop_path (float | Sequence[float]): DropPath / Stochastic Depth 比例
            norm_layer (nn.Module): 归一化层类型

        形状:
            输入:  x 形状为 (B, C, H, W)
            输出:  x 形状为 (B, middle_resolution[0] * middle_resolution[1], 2 * dim)
                  skip 形状为 (B, input_resolution[0], input_resolution[1], dim)

        Example:
            >>> encoder = FengWuEncoder(
            ...     input_resolution=(181, 360),
            ...     middle_resolution=(91, 180),
            ...     in_chans=37,
            ...     img_size=(721, 1440),
            ...     patch_size=(4, 4),
            ...     dim=192,
            ... )
            >>> x = torch.randn(2, 37, 721, 1440)
            >>> out, skip = encoder(x)
            >>> out.shape
            torch.Size([2, 91 * 180, 192 * 2])
            >>> skip.shape
            torch.Size([2, 181, 360, 192])
    """
    def __init__(
        self,
        input_resolution=(181, 360),
        middle_resolution=(91,180),
        in_chans=37,
        img_size=(721, 1440),
        patch_size=(4, 4),
        dim=192,
        depth=2,
        depth_middle=6,
        num_heads=(6, 12),
        window_size=(6, 12),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.depth_middle = depth_middle
        if isinstance(drop_path, Sequence):
            drop_path_middle = drop_path[depth:]
            drop_path = drop_path[:depth]
        else:
            drop_path_middle = drop_path
        if isinstance(num_heads, Sequence):
            num_heads_middle = num_heads[1]
            num_heads = num_heads[0]
        else:
            num_heads_middle = num_heads

        self.patchembed2d = OneEmbedding(
            style="PanguEmbedding2D",
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=dim,
        )
        self.blocks = nn.ModuleList(
            [   
                OneTransformer(
                    style="EarthTransformer2DBlock",
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=(0, 0) if i % 2 == 0 else None,
                    drop_path=drop_path[i] if isinstance(drop_path, Sequence) else drop_path,

                )
                for i in range(depth)
            ]
        )

        self.downsample = OneSample(
            style="PanguDownSample2D",
            in_dim=dim,
            input_resolution=input_resolution,
            output_resolution=middle_resolution,
        )

        self.blocks_middle = nn.ModuleList(
            [
                OneTransformer(
                    style="EarthTransformer2DBlock",
                    dim=dim * 2,
                    input_resolution=middle_resolution,
                    num_heads=num_heads_middle,
                    window_size=window_size,
                    shift_size=(0, 0) if i % 2 == 0 else None,
                    drop_path=drop_path_middle[i] if isinstance(drop_path_middle, Sequence) else drop_path_middle,
                )
                for i in range(depth_middle)
            ]
        )

    def forward(self, x):
        x = self.patchembed2d(x)
        B, C, Lat, Lon = x.shape
        x = x.reshape(B, C, -1).transpose(1, 2)
        for blk in self.blocks:
            x = blk(x)
        skip = x.reshape(B, Lat, Lon, C)
        x = self.downsample(x)
        for blk in self.blocks_middle:
            x = blk(x)
        return x, skip