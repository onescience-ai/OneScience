from collections.abc import Sequence

import torch
from timm.layers import to_2tuple
from timm.models.swin_transformer import SwinTransformerStage
from torch import nn
from ..func_utils import DropPath, Mlp, get_pad2d, crop2d, window_partition, window_reverse, get_shift_window_mask
from onescience.modules.attention.oneattention import OneAttention

class EarthTransformer2DBlock(nn.Module):
    """
    Revise from WeatherLearn https://github.com/lizhuoq/WeatherLearn
    2D Transformer Block
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size [latitude, longitude].
        shift_size (tuple[int]): Shift size for SW-MSA [latitude, longitude].
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
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