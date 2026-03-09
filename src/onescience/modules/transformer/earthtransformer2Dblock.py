from collections.abc import Sequence

import torch
from timm.layers import to_2tuple
from timm.models.swin_transformer import SwinTransformerStage
from torch import nn
from ..func_utils import DropPath, Mlp, get_pad2d, crop2d, window_partition, window_reverse, get_shift_window_mask
from onescience.modules.attention.oneattention import OneAttention

class EarthTransformer2DBlock(nn.Module):
    """
        用于2D地表变量的地球感知 Swin Transformer Block。
        
        标准 Swin Transformer Block 的气象场适配版本，结合 EarthAttention2D 实现
        带地球位置偏置的窗口注意力。支持循环移位（Shifted Window）以扩大感受野，
        并通过 ZeroPad + Crop 处理输入分辨率与窗口大小不整除的情况。
        每两个相邻 Block 交替使用普通窗口（shift_size=(0,0)）和移位窗口，
        构成完整的 W-MSA + SW-MSA 配对。
        
        Args:
            dim (int): 输入 token 的通道数（嵌入维度）。
            input_resolution (tuple[int, int]): 输入特征图的空间分辨率 (lat, lon)，
                不整除时模块内部自动 padding 后处理，输出分辨率与输入保持一致。
            num_heads (int): 多头注意力的头数。
            window_size (tuple[int, int], optional): 注意力窗口大小 (Wlat, Wlon)，
                默认为 (6, 12)。
            shift_size (tuple[int, int], optional): 循环移位的偏移量 (shift_lat, shift_lon)，
                设为 (0, 0) 时为普通窗口注意力（W-MSA），默认为 (3, 6) 即半窗口偏移
                （SW-MSA）。
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
            - 输入 x: (B, lat * lon, C)，其中 C = dim
            - 输出:   (B, lat * lon, C)，分辨率与通道数均不变
        
        Examples:
            >>> # W-MSA Block（偶数层，不做移位）
            >>> # 分辨率 181×360，窗口大小 6×12
            >>> # lat=181 不能被6整除，自动 padding 后处理
            >>> block_w = EarthTransformer2DBlock(
            ...     dim=192,
            ...     input_resolution=(181, 360),
            ...     num_heads=6,
            ...     window_size=(6, 12),
            ...     shift_size=(0, 0),  # 普通窗口
            ... )
            >>> x = torch.randn(2, 181 * 360, 192)  # (B, lat*lon, C)
            >>> out = block_w(x)
            >>> out.shape
            torch.Size([2, 65160, 192])
            
            >>> # SW-MSA Block（奇数层，移位半窗口）
            >>> block_sw = EarthTransformer2DBlock(
            ...     dim=192,
            ...     input_resolution=(181, 360),
            ...     num_heads=6,
            ...     window_size=(6, 12),
            ...     shift_size=(3, 6),  # 半窗口移位
            ... )
            >>> out = block_sw(x)
            >>> out.shape
            torch.Size([2, 65160, 192])
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
        window_size = (6, 12) if window_size is None else window_size
        shift_size = (3, 6) if shift_size is None else shift_size
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        padding = get_pad2d(input_resolution, window_size)
        self.pad = nn.ZeroPad2d(padding)

        pad_resolution = list(input_resolution)
        pad_resolution[0] += padding[2] + padding[3]
        pad_resolution[1] += padding[0] + padding[1]

        self.attn = OneAttention(
            style="EarthAttention2D",
            dim=dim,
            input_resolution=pad_resolution,
            window_size=window_size,
            num_heads=num_heads,
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

        shift_lat, shift_lon = self.shift_size
        self.roll = shift_lon and shift_lat

        if self.roll:
            attn_mask = get_shift_window_mask(
                pad_resolution, window_size, shift_size, ndim=2
            )
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor):
        Lat, Lon = self.input_resolution
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, Lat, Lon, C)

        # start pad
        x = self.pad(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        _, Lat_pad, Lon_pad, _ = x.shape

        shift_lat, shift_lon = self.shift_size
        if self.roll:
            shifted_x = torch.roll(x, shifts=(-shift_lat, -shift_lat), dims=(1, 2))
            x_windows = window_partition(shifted_x, self.window_size, ndim=2)
            # B*num_lon, num_lat, win_lat, win_lon, C
        else:
            shifted_x = x
            x_windows = window_partition(shifted_x, self.window_size, ndim=2)
            # B*num_lon, num_lat, win_lat, win_lon, C

        win_lat, win_lon = self.window_size
        x_windows = x_windows.view(
            x_windows.shape[0], x_windows.shape[1], win_lat * win_lon, C
        )
        # B*num_lon, num_lat, win_lat*win_lon, C

        attn_windows = self.attn(
            x_windows, mask=self.attn_mask
        )  # B*num_lon, num_lat, win_lat*win_lon, C

        attn_windows = attn_windows.view(
            attn_windows.shape[0], attn_windows.shape[1], win_lat, win_lon, C
        )

        if self.roll:
            shifted_x = window_reverse(
                attn_windows, self.window_size, Lat=Lat_pad, Lon=Lon_pad, ndim=2
            )
            # B * Lat * Lon * C
            x = torch.roll(shifted_x, shifts=(shift_lat, shift_lon), dims=(1, 2))
        else:
            shifted_x = window_reverse(
                attn_windows, self.window_size, Lat=Lat_pad, Lon=Lon_pad, ndim=2
            )
            x = shifted_x

        # crop, end pad
        x = crop2d(x.permute(0, 3, 1, 2), self.input_resolution).permute(0, 2, 3, 1)

        x = x.reshape(B, Lat * Lon, C)
        x = shortcut + self.drop_path(x)

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x