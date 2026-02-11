import torch

from torch import nn
from .xihetransformer import EarthAttention3D

class XiHeTransformer3D(nn.Module):
    """
    Revise from WeatherLearn https://github.com/lizhuoq/WeatherLearn
    3D Transformer Block
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size [pressure levels, latitude, longitude].
        shift_size (tuple[int]): Shift size for SW-MSA [pressure levels, latitude, longitude].
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
        attn_mask=None

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
        

    def forward(self, x: torch.Tensor,mask: torch.Tensor = None):
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
        else:        
            shifted_x = x
            x_windows = window_partition(shifted_x, self.window_size)
        win_pl, win_lat, win_lon = self.window_size
        
        x_windows = x_windows.view(
            x_windows.shape[0], x_windows.shape[1], win_pl * win_lat * win_lon, C
        )
 
        attn_mask = None
        if mask is not None:
            # 期望 mask 是 [B, 1, Lat, Lon] 或 [B, 1, Pl, Lat, Lon]
            if mask.dim() == 4:                # (B,1,Lat,Lon) -> (B,1,1,Lat,Lon)
                mask = mask.unsqueeze(2)

            # 此时 mask: (B, 1, Pl, Lat, Lon) 期望 (N, C, D, H, W)；这里 C=1, D=Pl, H=Lat, W=Lon，直接 pad 即可
            mask = self.pad(mask)              # (B, 1, Pl_pad, Lat_pad, Lon_pad)

            # 为了与 window_partition 通用实现对齐，转成 (B, Pl_pad, Lat_pad, Lon_pad, 1)
            mask5d = mask.permute(0, 2, 3, 4, 1).contiguous()

            # 与特征 x 完全一致的分块（3D窗口）
            # mwin: (B*num_lon, num_pl*num_lat, win_pl, win_lat, win_lon, 1)
            mwin = window_partition(mask5d, self.window_size)

            win_pl, win_lat, win_lon = self.window_size
            # 计算分块数量
            # 注意：x 已经 pad 过，这里的 Pl_pad/Lat_pad/Lon_pad 要和上面 x 的 pad 后维度一致
            _, Pl_pad, Lat_pad, Lon_pad, _ = x.shape               # x 此时是 pad 后的 (B, Pl_pad, Lat_pad, Lon_pad, C)
            B_eff  = mask5d.shape[0]
            num_lon   = Lon_pad // win_lon
            num_pllat = (Pl_pad // win_pl) * (Lat_pad // win_lat)
            N = win_pl * win_lat * win_lon                         # 每个窗口 token 数

            # 把 (B*num_lon, num_pl*num_lat, win_pl, win_lat, win_lon, 1) 还原出 (B, num_lon, num_pl*num_lat, N)
            mwin = mwin.view(B_eff, num_lon, num_pllat, win_pl, win_lat, win_lon, 1)
            # 取第 0 个 batch
            mwin = mwin[0]                                         # (num_lon, num_pl*num_lat, win_pl, win_lat, win_lon, 1)
            mwin = mwin.view(num_lon, num_pllat, N)                # (num_lon, num_pl*num_lat, N)，元素∈{0,1}

            # 生成注意力掩码 (num_lon, num_pl*num_lat, N, N) 仅允许 海×海，其他（涉及陆地）设为 -inf
            attn_mask = (mwin.unsqueeze(-1) * mwin.unsqueeze(-2))  # 0/1
            attn_mask = (attn_mask == 0).float() * -100.0          # 变成 0 / -100

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
        #两次残差
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))