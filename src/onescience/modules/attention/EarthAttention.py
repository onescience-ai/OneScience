import torch
from torch import nn


from ..func_utils.pangu_utils import (
    get_earth_position_index,
    trunc_normal_,
)

class EarthAttention3D(nn.Module):
    """
        改编自 WeatherLearn（https://github.com/lizhuoq/WeatherLearn）
        具有地球位置偏置（Earth Position Bias）的三维窗口注意力机制（3D Window Attention）。
        该模块同时支持 移位（shifted） 和 非移位（non-shifted） 窗口机制。

        参数说明：
        dim (int)：输入通道数。
        input_resolution (tuple[int])：输入的空间分辨率，格式为 [气压层数, 纬度, 经度]。
        window_size (tuple[int])：窗口大小，格式为 [气压层数, 纬度, 经度]。
        num_heads (int)：注意力头（Attention Heads）的数量。
        qkv_bias (bool, 可选)：如果为 True，则在 Query、Key、Value 上添加可学习偏置项。默认值：True。
        qk_scale (float | None, 可选)：如果设置该参数，则会覆盖默认的缩放比例（head_dim ** -0.5）。
        attn_drop (float, 可选)：注意力权重的 dropout 比例。默认值：0.0。
        proj_drop (float, 可选)：输出的 dropout 比例。默认值：0.0。
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
        """
        参数:
            x: 输入特征张量，形状为 (B * num_lon, num_pl*num_lat, N, C)
            mask: 取值为 0 或 -∞，形状为 (num_lon, num_pl*num_lat, Wpl*Wlat*Wlon, Wpl*Wlat*Wlon)
        """
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
    改编自 WeatherLearn https://github.com/lizhuoq/WeatherLearn
    功能描述：
        这是一个带有地球位置偏置（earth position bias）的二维窗口注意力（2D window attention）模块。
        该模块同时支持带位移（shifted）和不带位移（non-shifted）的窗口机制。
    参数:
        dim (int): 输入通道数。
        输入特征图的空间尺寸，格式为: [latitude, longitude]
        窗口大小，格式为: [latitude, longitude]
        num_heads (int): 注意力头（multi-head attention）的数量。
        qkv_bias (bool, optional):  若为True，则在Query、Key、Value向量中添加可学习的偏置项。默认值为True。
        qk_scale (float | None, optional): 若设置该值，则使用此值替代默认的缩放系数 head_dim ** -0.5 if set
        attn_drop (float, optional): 注意力权重的 dropout 比例。默认值为 0.0。
        proj_drop (float, optional): 注意力输出的 dropout 比例。默认值为 0.0
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
        """
        参数:
            x: 输入特征张量，形状为 (B * num_lon, num_lat, N, C)
            取值为 0 或 -∞，形状为 (num_lon, num_lat, Wlat*Wlon, Wlat*Wlon)
        """
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
