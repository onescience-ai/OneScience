from collections.abc import Sequence
import torch
import torch.nn as nn
from onescience.modules.func_utils import (
    Mlp, DistributedMlp, crop3d, get_pad3d,
    window_partition, window_reverse, DropPath,
)
from onescience.modules.attention.oneattention import OneAttention


class XiheDistributedLocalTransformer(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        num_heads=6,
        window_size=(1, 6, 12),
        shift_size=(0, 0, 0),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        config=None,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        padding = get_pad3d(input_resolution, window_size)
        self.pad = nn.ZeroPad3d(padding)
        attn_mask = None

        pad_resolution = list(input_resolution)
        pad_resolution[0] += padding[-1] + padding[-2]
        pad_resolution[1] += padding[2] + padding[3]
        pad_resolution[2] += padding[0] + padding[1]

        self.attn = OneAttention(
            style="EarthDistributedAttention3D",
            dim=dim,
            input_resolution=pad_resolution,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            config=config,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = DistributedMlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            config=config,
        )

        shift_pl, shift_lat, shift_lon = self.shift_size
        self.roll = shift_pl and shift_lon and shift_lat

        self.register_buffer("attn_mask", None)

    def forward(self, obj):
        if isinstance(obj, dict):
            x = obj["x"]
            mask = obj.get("mask")
            if mask is not None:
                mask = mask.clone().detach().float()
        else:
            x = obj.x
            mask = getattr(obj, 'mask', None)
            obj = {"x": x, "mask": mask}

        Pl, Lat, Lon = self.input_resolution
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, Pl, Lat, Lon, C)

        x = self.pad(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)

        _, Pl_pad, Lat_pad, Lon_pad, _ = x.shape

        shift_pl, shift_lat, shift_lon = self.shift_size

        if self.roll:
            shifted_x = torch.roll(
                x, shifts=(-shift_pl, -shift_lat, -shift_lon), dims=(1, 2, 3)
            )
            x_windows = window_partition(shifted_x, self.window_size)
        else:
            shifted_x = x
            x_windows = window_partition(shifted_x, self.window_size)

        win_pl, win_lat, win_lon = self.window_size
        x_windows = x_windows.view(
            x_windows.shape[0], x_windows.shape[1], win_pl * win_lat * win_lon, C
        )

        attn_mask = None
        if mask is not None:
            if mask.dim() == 4:
                mask = mask.unsqueeze(2)
            mask = self.pad(mask)
            mask5d = mask.permute(0, 2, 3, 4, 1).contiguous()
            mwin = window_partition(mask5d, self.window_size)

            win_pl, win_lat, win_lon = self.window_size
            _, Pl_pad, Lat_pad, Lon_pad, _ = x.shape
            B_eff = mask5d.shape[0]
            num_lon = Lon_pad // win_lon
            num_pllat = (Pl_pad // win_pl) * (Lat_pad // win_lat)
            N = win_pl * win_lat * win_lon

            mwin = mwin.view(B_eff, num_lon, num_pllat, win_pl, win_lat, win_lon, 1)
            mwin = mwin[0]
            mwin = mwin.view(num_lon, num_pllat, N)

            attn_mask = (mwin.unsqueeze(-1) * mwin.unsqueeze(-2))
            attn_mask = (attn_mask == 0).float() * -100.0

        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(
            attn_windows.shape[0], attn_windows.shape[1], win_pl, win_lat, win_lon, C
        )

        if self.roll:
            shifted_x = window_reverse(
                attn_windows, self.window_size, Pl=Pl_pad, Lat=Lat_pad, Lon=Lon_pad
            )
            x = torch.roll(
                shifted_x, shifts=(shift_pl, shift_lat, shift_lon), dims=(1, 2, 3)
            )
        else:
            shifted_x = window_reverse(
                attn_windows, self.window_size, Pl=Pl_pad, Lat=Lat_pad, Lon=Lon_pad
            )
            x = shifted_x

        x = crop3d(x.permute(0, 4, 1, 2, 3), self.input_resolution).permute(
            0, 2, 3, 4, 1
        )

        x = x.reshape(B, Pl * Lat * Lon, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
