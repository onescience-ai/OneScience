from collections.abc import Sequence

import torch
from timm.layers import to_2tuple
from timm.models.swin_transformer import SwinTransformerStage
from torch import nn
from ..func_utils import DropPath, Mlp, get_pad3d, crop3d, window_partition, window_reverse, get_shift_window_mask

from onescience.modules.transformer.onetransformer import OneTransformer


class PanguFuser(nn.Module):
    """
        Pangu-Weather模型的三维特征融合模块。

        Pangu-Weather在主干编码与解码阶段，使用该模块在给定的三维网格上堆叠多层
        3D Transformer 块，对 patch token 进行层次化特征交互。

        该模块接收已经完成 patch embedding、并按统一三维网格组织后的 token 序列。
        在 Pangu 主模型中，这些 token 通常由：
        - 地表分支的二维 patch 特征
        - 高空分支的三维 patch 特征
        沿 PressureLevels 维拼接后得到。

        因此，该模块负责在 `(PressureLevels, Height, Width)` 三维网格上融合：
        - 不同气压层之间的信息
        - 局部空间邻域的信息
        - 多层 Transformer block 逐步建模后的层次特征

        Args:
            dim (int):
                输入与输出 token 的特征维度。
            input_resolution (tuple[int, int, int]):
                三维输入特征的网格尺寸 `(PressureLevels, Height, Width)`。
            depth (int):
                `EarthTransformer3DBlock` 的堆叠层数。
            num_heads (int):
                多头自注意力的头数。
            window_size (tuple[int, int, int]):
                三维窗口注意力的窗口大小 `(Pl_window, H_window, W_window)`。
            drop_path (float | Sequence[float]):
                DropPath / Stochastic Depth 比例或其序列。
            mlp_ratio (float):
                前馈网络隐藏层相对特征维度的放大比例。
            qkv_bias (bool):
                是否在 QKV 投影中使用偏置。
            qk_scale (float | None):
                QK 点积缩放因子。
            drop (float):
                特征上的 dropout 比例。
            attn_drop (float):
                注意力权重上的 dropout 比例。
            norm_layer (nn.Module):
                归一化层类型。

        形状:
            输入:
                `x` 形状为 `(Batch, PressureLevels * Height * Width, dim)`，
                其中 `(PressureLevels, Height, Width) = input_resolution`
            输出:
                `x` 形状为 `(Batch, PressureLevels * Height * Width, dim)`，与输入相同

        Example:
            >>> # Pangu-Weather 主干中的第一层特征融合
            >>> dim = 192
            >>> input_resolution = (8, 181, 360)
            >>> fuser = PanguFuser(
            ...     dim=dim,
            ...     input_resolution=input_resolution,
            ...     depth=2,
            ...     num_heads=6,
            ...     window_size=(2, 6, 12),
            ... )
            >>> Batch = 2
            >>> PressureLevels = 8
            >>> Height = 181
            >>> Width = 360
            >>> x = torch.randn(Batch, PressureLevels * Height * Width, dim)
            >>> out = fuser(x)
            >>> out.shape
            torch.Size([2, 8 * 181 * 360, 192])
    """
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        drop_path=0.0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,  
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        self.blocks = nn.ModuleList(
            [
                OneTransformer(
                    style="EarthTransformer3DBlock",
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=(0, 0, 0) if i % 2 == 0 else None,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, Sequence)
                    else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x
