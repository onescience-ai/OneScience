import torch
from torch import nn

from ..func_utils import (
    get_earth_position_index,
    trunc_normal_,
)

class EarthAttention2D(nn.Module):
    """
        用于2D地表变量的地球位置偏置窗口注意力机制。
        
        在标准窗口注意力基础上引入地球位置偏置（Earth Position Bias），
        用于捕捉纬度方向窗口间的空间关系。适用于处理地表气象变量（如
        2m温度、10m风速等）的Swin-Transformer风格注意力计算。
        
        Args:
            dim (int): 输入通道数（嵌入维度）。
            input_resolution (tuple[int, int]): 输入特征图的空间分辨率 (lat, lon)，
                用于计算纬度方向的窗口数量 num_lat = lat // Wlat。
            window_size (tuple[int, int]): 注意力窗口大小 (Wlat, Wlon)。
            num_heads (int): 多头注意力的头数。
            qkv_bias (bool, optional): 是否为QKV投影添加偏置项，默认为True。
            qk_scale (float, optional): QK点积的缩放系数，默认为None，
                此时自动使用 head_dim ** -0.5。
            attn_drop (float, optional): 注意力权重的Dropout比例，默认为0.0。
            proj_drop (float, optional): 输出投影的Dropout比例，默认为0.0。
        
        形状:
            - 输入 x: (B * num_lon, num_lat, Wlat * Wlon, C)
                其中 num_lat = lat // Wlat，num_lon = lon // Wlon
            - 输入 mask: (num_lon, num_lat, Wlat * Wlon, Wlat * Wlon) 或 None
            - 输出: (B * num_lon, num_lat, Wlat * Wlon, C)
        
        Examples:
            >>> # 典型Pangu-Weather地表变量配置
            >>> # 分辨率: lat=128, lon=256，窗口大小: 8×8
            >>> # num_lat = 128 // 8 = 16
            >>> # num_lon = 256 // 8 = 32
            >>> # B_ = B * num_lon = 4 * 32 = 128
            >>> # N = Wlat * Wlon = 8 * 8 = 64
            >>> attn = EarthAttention2D(
            ...     dim=192,
            ...     input_resolution=(128, 256),
            ...     window_size=(8, 8),
            ...     num_heads=6,
            ... )
            >>> x = torch.randn(128, 16, 64, 192)  # (B*num_lon, num_lat, N, C)
            >>> out = attn(x)
            >>> out.shape
            torch.Size([128, 16, 64, 192])
            
            >>> # 带mask的前向传播（用于循环边界填充场景）
            >>> mask = torch.zeros(32, 16, 64, 64)  # (num_lon, num_lat, N, N)
            >>> out = attn(x, mask=mask)
            >>> out.shape
            torch.Size([128, 16, 64, 192])
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
        Args:
            x: input features with shape of (B * num_lon, num_lat, N, C)
            mask: (0/-inf) mask with shape of (num_lon, num_lat, Wlat*Wlon, Wlat*Wlon)
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
