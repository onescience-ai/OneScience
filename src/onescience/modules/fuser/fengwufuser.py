from collections.abc import Sequence

import torch
from timm.layers import to_2tuple
from timm.models.swin_transformer import SwinTransformerStage
from torch import nn
from onescience.modules.transformer.onetransformer import OneTransformer


class FengWuFuser(nn.Module):
    """
        FengWu 模型的三维特征融合模块，在中分辨率的时空网格上堆叠多层 3D Transformer 块以融合多时刻、多变量信息。

        Args:
            input_resolution (tuple[int, int, int]): 三维输入特征的网格尺寸 (T, H, W)
            dim (int): 输入与输出特征的通道维度
            depth (int): 3D Transformer 块的层数
            num_heads (int): 多头自注意力的头数
            window_size (tuple[int, int, int]): 三维窗口注意力的窗口大小 (Wt, Wh, Ww)
            mlp_ratio (float): 前馈网络隐藏层与特征维度的比例
            qkv_bias (bool): 是否在 QKV 投影中使用偏置
            qk_scale (float | None): QK 点积缩放因子
            drop (float): 特征上的 dropout 比例
            attn_drop (float): 注意力权重上的 dropout 比例
            drop_path (float | Sequence[float]): DropPath / Stochastic Depth 比例或其序列
            norm_layer (nn.Module): 归一化层类型

        形状:
            输入:  x 形状为 (B, T * H * W, dim)，其中 (T, H, W) = input_resolution
            输出:  x 形状与输入相同，为 (B, T * H * W, dim)

        Example:
            >>> fuser = FengWuFuser(
            ...     input_resolution=(6, 91, 180),
            ...     dim=192 * 2,
            ...     depth=6,
            ...     num_heads=12,
            ...     window_size=(2, 6, 12),
            ... )
            >>> B, T, H, W, C = 2, 6, 91, 180, 192 * 2
            >>> x = torch.randn(B, T * H * W, C)  # 已展平的三维网格特征
            >>> out = fuser(x)
            >>> out.shape
            torch.Size([2, T * H * W, C])
    """
    def __init__(
        self,
        input_resolution=(6, 91, 180),
        dim=192*2,
        depth=6,
        num_heads=12,
        window_size=(2, 6, 12),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=[0.2] * 6,
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
                    drop_path=drop_path[i] if isinstance(drop_path, Sequence) else drop_path,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x