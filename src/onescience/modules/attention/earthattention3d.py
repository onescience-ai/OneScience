import torch
from torch import nn

from ..func_utils import (
    get_earth_position_index,
    trunc_normal_,
)

class EarthAttention3D(nn.Module):
    """
        用于3D大气变量的地球位置偏置窗口注意力机制。
        
        EarthAttention2D 的三维扩展版本，在气压层（pressure level）维度上
        额外引入位置偏置，用于同时捕捉垂直方向与水平方向的空间关系。适用于
        处理多层大气变量（如位势高度、温度、风场等等压面数据）的注意力计算。
        
        Args:
            dim (int): 输入通道数（嵌入维度）。
            input_resolution (tuple[int, int, int]): 输入特征图的空间分辨率
                (pl, lat, lon)，用于计算窗口数量：
                - num_pl = pl // Wpl
                - num_lat = lat // Wlat
                其中 type_of_windows = num_pl * num_lat，经度方向折叠进 batch 维。
            window_size (tuple[int, int, int]): 注意力窗口大小 (Wpl, Wlat, Wlon)。
            num_heads (int): 多头注意力的头数。
            qkv_bias (bool, optional): 是否为QKV投影添加偏置项，默认为True。
            qk_scale (float, optional): QK点积的缩放系数，默认为None，
                此时自动使用 head_dim ** -0.5。
            attn_drop (float, optional): 注意力权重的Dropout比例，默认为0.0。
            proj_drop (float, optional): 输出投影的Dropout比例，默认为0.0。
        
        形状:
            - 输入 x: (B * num_lon, num_pl * num_lat, Wpl * Wlat * Wlon, C)
                其中 num_pl = pl // Wpl，num_lat = lat // Wlat，num_lon = lon // Wlon
            - 输入 mask: (num_lon, num_pl * num_lat, Wpl * Wlat * Wlon, Wpl * Wlat * Wlon) 或 None
            - 输出: (B * num_lon, num_pl * num_lat, Wpl * Wlat * Wlon, C)
        
            Examples:
            >>> # 典型Pangu-Weather大气变量配置
            >>> # 原始气压层数为13，经 get_pad3d padding 后 pl=14（对齐 Wpl=2）
            >>> # pad_resolution = (14, 128, 256)，window_size = (2, 8, 8)
            >>> # num_pl  = 14 // 2 = 7
            >>> # num_lat = 128 // 8 = 16
            >>> # num_lon = 256 // 8 = 32
            >>> # type_of_windows = num_pl * num_lat = 7 * 16 = 112
            >>> # B_ = B * num_lon = 4 * 32 = 128
            >>> # N = Wpl * Wlat * Wlon = 2 * 8 * 8 = 128
            >>> attn = EarthAttention3D(
            ...     dim=192,
            ...     input_resolution=(14, 128, 256),  # 传入 padding 后的分辨率
            ...     window_size=(2, 8, 8),
            ...     num_heads=6,
            ... )
            >>> x = torch.randn(128, 112, 128, 192)  # (B*num_lon, num_pl*num_lat, N, C)
            >>> out = attn(x)
            >>> out.shape
            torch.Size([128, 112, 128, 192])
            
            >>> # 带mask的前向传播（经度循环边界填充场景）
            >>> mask = torch.zeros(32, 112, 128, 128)  # (num_lon, num_pl*num_lat, N, N)
            >>> out = attn(x, mask=mask)
            >>> out.shape
            torch.Size([128, 112, 128, 192])
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