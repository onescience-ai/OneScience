from collections.abc import Sequence

import torch
from timm.layers import to_2tuple
from timm.models.swin_transformer import SwinTransformerStage
from torch import nn
from ..utils import DropPath, Mlp, get_pad3d, crop3d, window_partition, window_reverse, get_shift_window_mask

from onescience.modules import EarthAttention3D

class Transformer3DBlock(nn.Module):
    """
    改编自 WeatherLearn 项目 https://github.com/lizhuoq/WeatherLearn
    3D Transformer Block
    参数:
        dim (int): 输入特征的通道数。
        input_resolution (tuple[int]): 输入特征的空间分辨率。
        num_heads (int): 多头注意力的头数。
        window_size (tuple[int]): 局部注意力窗口大小， [pressure levels, latitude, longitude].
        shift_size (tuple[int]): 在 “Shifted Window Multi-Head Self-Attention (SW-MSA)” 模式下的窗口平移大小 [pressure levels, latitude, longitude].
        mlp_ratio (float): MLP 隐藏层维度与嵌入维度的比例。
        qkv_bias (bool, optional): 若为 True，则在 Query、Key、Value 的线性层中添加可学习偏置项（bias）。默认 True。
        qk_scale (float | None, optional): 若设置该值，则覆盖默认的缩放因子 head_dim ** -0.5。通常用于调整注意力计算的数值范围。
        drop (float, optional): Dropout 比例，作用于前向传播中的全连接层输出。默认 0.0。
        attn_drop (float, optional): 注意力权重的 Dropout 比例。默认 0.0。
        drop_path (float, optional): 随机深度（Stochastic Depth）比例，用于防止过拟合。默认 0.0。
        act_layer (nn.Module, optional): 激活函数层，默认使用 nn.GELU。
        norm_layer (nn.Module, optional): 归一化层类型，默认使用 nn.LayerNorm。
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=None,
        shift_size=None,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        window_size = (2, 6, 12) if window_size is None else window_size
        shift_size = (1, 3, 6) if shift_size is None else shift_size
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        padding = get_pad3d(input_resolution, window_size)
        self.pad = nn.ZeroPad3d(padding)

        pad_resolution = list(input_resolution)
        pad_resolution[0] += padding[-1] + padding[-2]
        pad_resolution[1] += padding[2] + padding[3]
        pad_resolution[2] += padding[0] + padding[1]

        self.attn = EarthAttention3D(
            dim=dim,
            input_resolution=pad_resolution,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        shift_pl, shift_lat, shift_lon = self.shift_size
        self.roll = shift_pl and shift_lon and shift_lat

        if self.roll:
            attn_mask = get_shift_window_mask(pad_resolution, window_size, shift_size)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor):
        Pl, Lat, Lon = self.input_resolution
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, Pl, Lat, Lon, C)

        # start pad
        x = self.pad(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)

        _, Pl_pad, Lat_pad, Lon_pad, _ = x.shape

        shift_pl, shift_lat, shift_lon = self.shift_size
        if self.roll:
            shifted_x = torch.roll(
                x, shifts=(-shift_pl, -shift_lat, -shift_lat), dims=(1, 2, 3)
            )
            x_windows = window_partition(shifted_x, self.window_size)
            # B*num_lon, num_pl*num_lat, win_pl, win_lat, win_lon, C
        else:
            shifted_x = x
            x_windows = window_partition(shifted_x, self.window_size)
            # B*num_lon, num_pl*num_lat, win_pl, win_lat, win_lon, C

        win_pl, win_lat, win_lon = self.window_size
        x_windows = x_windows.view(
            x_windows.shape[0], x_windows.shape[1], win_pl * win_lat * win_lon, C
        )
        # B*num_lon, num_pl*num_lat, win_pl*win_lat*win_lon, C

        attn_windows = self.attn(
            x_windows, mask=self.attn_mask
        )  # B*num_lon, num_pl*num_lat, win_pl*win_lat*win_lon, C

        attn_windows = attn_windows.view(
            attn_windows.shape[0], attn_windows.shape[1], win_pl, win_lat, win_lon, C
        )

        if self.roll:
            shifted_x = window_reverse(
                attn_windows, self.window_size, Pl=Pl_pad, Lat=Lat_pad, Lon=Lon_pad
            )
            # B * Pl * Lat * Lon * C
            x = torch.roll(
                shifted_x, shifts=(shift_pl, shift_lat, shift_lon), dims=(1, 2, 3)
            )
        else:
            shifted_x = window_reverse(
                attn_windows, self.window_size, Pl=Pl_pad, Lat=Lat_pad, Lon=Lon_pad
            )
            x = shifted_x

        # crop, end pad
        x = crop3d(x.permute(0, 4, 1, 2, 3), self.input_resolution).permute(
            0, 2, 3, 4, 1
        )

        x = x.reshape(B, Pl * Lat * Lon, C)
        x = shortcut + self.drop_path(x)

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class FuserLayer(nn.Module):
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
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        self.blocks = nn.ModuleList(
            [
                Transformer3DBlock(
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