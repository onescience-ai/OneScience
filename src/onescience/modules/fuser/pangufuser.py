from collections.abc import Sequence

import torch
from timm.layers import to_2tuple
from timm.models.swin_transformer import SwinTransformerStage
from torch import nn
from ..func_utils import DropPath, Mlp, get_pad3d, crop3d, window_partition, window_reverse, get_shift_window_mask

from onescience.modules.transformer.onetransformer import OneTransformer


class PanguFuser(nn.Module):
    """
    改编自 WeatherLearn 项目 https://github.com/lizhuoq/WeatherLearn
    一个阶段（stage）的基础3D Transformer层（Basic 3D Transformer Layer）

    参数:
        dim (int): 输入特征的通道数.
        input_resolution (tuple[int]): 输入数据的分辨率.
        depth (int): 当前阶段包含的 Transformer Block 数量.
        num_heads (int): 多头注意力（Multi-Head Attention）的头数.
        window_size (tuple[int]): 局部窗口的大小.
        mlp_ratio (float): MLP 隐藏层的维度与输入嵌入维度的比例.
        qkv_bias (bool, optional): 若为 True，则在 Query、Key、Value 的线性层中添加可学习偏置项。默认值：True
        qk_scale (float | None, optional): 若指定该值，则覆盖默认缩放因子 head_dim ** -0.5，用于调整注意力分数的缩放.
        drop (float, optional): Dropout 比例，用于防止过拟合。默认值：0.0
        attn_drop (float, optional): 注意力权重的 Dropout 比例。默认值：0.0
        drop_path (float | tuple[float], optional): 随机深度（Stochastic Depth）比例，可为单个数或一个不同层对应不同值的元组。默认值：0.0
        norm_layer (nn.Module, optional): 归一化层类型，默认使用 nn.LayerNorm
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