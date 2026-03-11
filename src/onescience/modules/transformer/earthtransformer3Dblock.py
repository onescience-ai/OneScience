from collections.abc import Sequence

import torch
from timm.layers import to_2tuple
from timm.models.swin_transformer import SwinTransformerStage
from torch import nn
from ..func_utils import DropPath, Mlp, get_pad3d, crop3d, window_partition, window_reverse, get_shift_window_mask

from ..attention.oneattention import OneAttention

class EarthTransformer3DBlock(nn.Module):
    """
        用于3D大气变量的地球感知 Swin Transformer Block。

        EarthTransformer2DBlock 的三维扩展版本，结合 EarthAttention3D 在气压层、
        纬度、经度三个维度上同时进行窗口注意力计算。支持三维循环移位（Shifted Window）
        以扩大跨窗口感受野，并通过 ZeroPad3d + crop3d 处理各维度与窗口大小不整除
        的情况。每两个相邻 Block 交替使用普通窗口（shift_size=(0,0,0)）和移位窗口，
        构成完整的 W-MSA + SW-MSA 配对。

        Args:
            dim (int): 输入 token 的通道数（嵌入维度）。
            input_resolution (tuple[int, int, int]): 输入特征图的空间分辨率 (pl, lat, lon)，
                不整除时模块内部自动 padding 后处理，输出分辨率与输入保持一致。
            num_heads (int): 多头注意力的头数。
            window_size (tuple[int, int, int], optional): 注意力窗口大小 (Wpl, Wlat, Wlon)，
                默认为 (2, 6, 12)。
            shift_size (tuple[int, int, int], optional): 循环移位的偏移量
                (shift_pl, shift_lat, shift_lon)，设为 (0, 0, 0) 时为普通窗口注意力
                （W-MSA），默认为 (1, 3, 6) 即各维度半窗口偏移（SW-MSA）。
            mlp_ratio (float, optional): MLP 隐层相对于 dim 的扩展倍数，默认为 4.0。
            qkv_bias (bool, optional): 是否为 QKV 投影添加偏置项，默认为 True。
            qk_scale (float, optional): QK 点积的缩放系数，默认为 None，
                自动使用 head_dim ** -0.5。
            drop (float, optional): MLP 的 Dropout 比例，默认为 0.0。
            attn_drop (float, optional): 注意力权重的 Dropout 比例，默认为 0.0。
            drop_path (float, optional): Stochastic Depth 的比例，默认为 0.0。
            act_layer (nn.Module, optional): MLP 的激活函数，默认为 nn.GELU。
            norm_layer (nn.Module, optional): 归一化层类型，默认为 nn.LayerNorm。

        形状:
            - 输入 x: (B, pl * lat * lon, C)，其中 C = dim
            - 输出:   (B, pl * lat * lon, C)，分辨率与通道数均不变

        Examples:
            >>> # W-MSA Block（偶数层，不做移位）
            >>> # 气压层13经 padding 对齐为14（Wpl=2），lat=128，lon=256
            >>> block_w = EarthTransformer3DBlock(
            ...     dim=192,
            ...     input_resolution=(13, 128, 256),
            ...     num_heads=6,
            ...     window_size=(2, 6, 12),
            ...     shift_size=(0, 0, 0),   # 普通窗口
            ... )
            >>> x = torch.randn(2, 13 * 128 * 256, 192)  # (B, pl*lat*lon, C)
            >>> out = block_w(x)
            >>> out.shape
            torch.Size([2, 425984, 192])

            >>> # SW-MSA Block（奇数层，三维半窗口移位）
            >>> block_sw = EarthTransformer3DBlock(
            ...     dim=192,
            ...     input_resolution=(13, 128, 256),
            ...     num_heads=6,
            ...     window_size=(2, 6, 12),
            ...     shift_size=(1, 3, 6),   # 半窗口移位
            ... )
            >>> out = block_sw(x)
            >>> out.shape
            torch.Size([2, 425984, 192])
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

        self.attn = OneAttention(
            style="EarthAttention3D",
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
                x, shifts=(-shift_pl, -shift_lat, -shift_lon), dims=(1, 2, 3)
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