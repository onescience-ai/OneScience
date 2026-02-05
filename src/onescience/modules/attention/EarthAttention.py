import torch
from torch import nn


from onescience.modules.func_utils.pangu_utils import (
    get_earth_position_index,
    trunc_normal_,
)

class EarthAttention3D(nn.Module):
    """
        具有地球位置偏置的三维窗口注意力机制。

        Args:
            dim (int) : 输入通道数
            input_resolution (*tuple[int, int, int]): 输入空间分辨率 (pressure_levels, lat, lon)
            window_size (tuple[int, int, int]): 窗口大小 (Wpl, Wlat, Wlon)
            num_heads (int): 注意力头数量
            qkv_bias (bool, optional): 是否在 QKV 上添加偏置，默认为 True
            qk_scale (float, optional): 覆盖默认的 QK 缩放系数 (head_dim ** -0.5)，默认为 None
            attn_drop (float, optional): 注意力权重的 dropout 比例，默认为 0.0
            proj_drop (float, optional): 输出的 dropout 比例，默认为 0.0

        形状:
            输入 x: (B × num_lon, num_pl × num_lat, N, C)，其中 N = Wpl × Wlat × Wlon
            输入 mask (可选): (num_lon, num_pl × num_lat, N, N)，值为 0 或 -∞
            输出: (B × num_lon, num_pl × num_lat, N, C)

        Example:
            >>> attn = EarthAttention3D(
            ...     dim=192,
            ...     input_resolution=(13, 128, 256), 
            ...     window_size=(1, 8, 8),
            ...     num_heads=6
            ... )
            >>> # num_pl=pressure_levels // window_size_1 = 13//1=13
            >>> # num_lat=lat // window_size_2 = 128//8=16
            >>> # num_lon=lon // window_size_3 = 256//8=32
            >>> # 输入: (B*num_lon, num_pl*num_lat, Wpl*Wlat*Wlon, C)
            >>> #      = (4*32, 13*16, 1*8*8, 192)
            >>> #      = (128, 208, 64, 192)
            >>> x = torch.randn(128, 208, 64, 192)
            >>> out = attn(x)
            >>> out.shape
            torch.Size([128, 208, 64, 192])

    """

    def __init__(
        self,
        dim,
        input_resolution,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wpl, Wlat, Wlon
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.type_of_windows = (input_resolution[0] // window_size[0]) * (
            input_resolution[1] // window_size[1]
        )

        self.earth_position_bias_table = nn.Parameter(
            torch.zeros(
                (window_size[0] ** 2)
                * (window_size[1] ** 2)
                * (window_size[2] * 2 - 1),
                self.type_of_windows,
                num_heads,
            )
        )  # Wpl**2 * Wlat**2 * Wlon*2-1, Npl//Wpl * Nlat//Wlat, nH

        earth_position_index = get_earth_position_index(
            window_size
        )  # Wpl*Wlat*Wlon, Wpl*Wlat*Wlon
        self.register_buffer("earth_position_index", earth_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.earth_position_bias_table = trunc_normal_(
            self.earth_position_bias_table, std=0.02
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask=None):
        B_, nW_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, nW_, N, 3, self.num_heads, C // self.num_heads)
            .permute(3, 0, 4, 1, 2, 5)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        earth_position_bias = self.earth_position_bias_table[
            self.earth_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.type_of_windows,
            -1,
        )  # Wpl*Wlat*Wlon, Wpl*Wlat*Wlon, num_pl*num_lat, nH
        earth_position_bias = earth_position_bias.permute(
            3, 2, 0, 1
        ).contiguous()  # nH, num_pl*num_lat, Wpl*Wlat*Wlon, Wpl*Wlat*Wlon
        attn = attn + earth_position_bias.unsqueeze(0)

        if mask is not None:
            nLon = mask.shape[0]
            attn = attn.view(
                B_ // nLon, nLon, self.num_heads, nW_, N, N
            ) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, nW_, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 2, 3, 1, 4).reshape(B_, nW_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EarthAttention2D(nn.Module):
    """
        具有地球位置偏置的二维窗口注意力机制。

        Args:
            dim (int): 输入通道数
            input_resolution (tuple[int, int]): 输入空间分辨率 (lat, lon)
            window_size (tuple[int, int]): 窗口大小 (Wlat, Wlon)
            num_heads (int): 注意力头数量
            qkv_bias (bool, optional): 是否在 QKV 上添加偏置，默认为 True
            qk_scale (float, optional): 覆盖默认的 QK 缩放系数，默认为 None
            attn_drop (float, optional): 注意力权重的 dropout 比例，默认为 0.0
            proj_drop (float, optional): 输出的 dropout 比例，默认为 0.0

        形状:
            输入 x: (B × num_lon, num_lat, N, C)，其中 N = Wlat × Wlon
            输入 mask (可选): (num_lon, num_lat, N, N)，值为 0 或 -∞
            输出: (B × num_lon, num_lat, N, C)

        Example:
            >>> # 地表变量的注意力计算
            >>> attn = EarthAttention2D(
            ...     dim=128,
            ...     input_resolution=(128, 256),
            ...     window_size=(8, 8),
            ...     num_heads=4
            ... )
            >>> # num_lat=lat // window_size_2 = 128//8=16
            >>> # num_lon=lon // window_size_3 = 256//8=32
            >>> # 输入: (B*num_lon, num_lat, Wlat*Wlon, C)
            >>> #      = (4*32, 16, 8*8, 128)
            >>> #      = (128, 16, 64, 128)
            >>> x = torch.randn(128, 16, 64, 128)
            >>> out = attn(x)
            >>> out.shape
            torch.Size([128, 16, 64, 128])

    """

    def __init__(
        self,
        dim,
        input_resolution,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wlat, Wlon
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.type_of_windows = input_resolution[0] // window_size[0]

        self.earth_position_bias_table = nn.Parameter(
            torch.zeros(
                (window_size[0] ** 2) * (window_size[1] * 2 - 1),
                self.type_of_windows,
                num_heads,
            )
        )  # Wlat**2 * Wlon*2-1, Nlat//Wlat, nH

        earth_position_index = get_earth_position_index(
            window_size, ndim=2
        )  # Wlat*Wlon, Wlat*Wlon
        self.register_buffer("earth_position_index", earth_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.earth_position_bias_table = trunc_normal_(
            self.earth_position_bias_table, std=0.02
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask=None):
        B_, nW_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, nW_, N, 3, self.num_heads, C // self.num_heads)
            .permute(3, 0, 4, 1, 2, 5)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        earth_position_bias = self.earth_position_bias_table[
            self.earth_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            self.type_of_windows,
            -1,
        )  # Wlat*Wlon, Wlat*Wlon, num_lat, nH
        earth_position_bias = earth_position_bias.permute(
            3, 2, 0, 1
        ).contiguous()  # nH, num_lat, Wlat*Wlon, Wlat*Wlon
        attn = attn + earth_position_bias.unsqueeze(0)

        if mask is not None:
            nLon = mask.shape[0]
            attn = attn.view(
                B_ // nLon, nLon, self.num_heads, nW_, N, N
            ) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, nW_, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 2, 3, 1, 4).reshape(B_, nW_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x