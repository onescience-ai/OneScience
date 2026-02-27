import torch
from torch import nn
import warnings

from onescience.modules.func_utils.pangu_utils import (
    get_earth_position_index,
    trunc_normal_,
)


class OneEarthAttention:
    """
    统一的Earth Attention接口，支持多种实现风格。
    
    具有地球位置偏置的窗口注意力机制，支持2D/3D数据。
    
    Args:
        dim (int): 输入通道数
        input_resolution (tuple): 输入空间分辨率
            - 2D: (lat, lon)
            - 3D: (pressure_levels, lat, lon)
        window_size (tuple): 窗口大小，形状需与input_resolution匹配
            - 2D: (Wlat, Wlon)
            - 3D: (Wpl, Wlat, Wlon)
        num_heads (int): 注意力头数量
        qkv_bias (bool, optional): 是否在QKV上添加偏置，默认为True
        qk_scale (float, optional): 覆盖默认的QK缩放系数，默认为None
        attn_drop (float, optional): 注意力权重的dropout比例，默认为0.0
        proj_drop (float, optional): 输出的dropout比例，默认为0.0
        style (str): 注意力实现风格，默认'pangu'
            可选值: ['pangu']
        **kwargs: 各style特定参数（当前pangu不需要额外参数）
    
    Examples:
        >>> # 2D地表变量注意力
        >>> attn = OneEarthAttention(
        ...     dim=128,
        ...     input_resolution=(128, 256),
        ...     window_size=(8, 8),
        ...     num_heads=4,
        ...     style='pangu'
        ... )
        
        >>> # 3D大气变量注意力
        >>> attn = OneEarthAttention(
        ...     dim=192,
        ...     input_resolution=(13, 128, 256),
        ...     window_size=(1, 8, 8),
        ...     num_heads=6,
        ...     style='pangu'
        ... )
    """
    
    _registry = {}
    
    def __new__(cls, dim, input_resolution, window_size, num_heads,
                qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0,
                style='pangu', **kwargs):
        if style not in cls._registry:
            available_styles = list(cls._registry.keys())
            warnings.warn(
                f"Style '{style}' not available. Available styles: {available_styles}. "
                f"Using 'pangu' as fallback.",
                UserWarning
            )
            style = 'pangu'
        
        return cls._registry[style](
            dim, input_resolution, window_size, num_heads,
            qkv_bias, qk_scale, attn_drop, proj_drop, **kwargs
        )
    
    @classmethod
    def register(cls, name):
        def wrapper(earth_attention_class):
            cls._registry[name] = earth_attention_class
            return earth_attention_class
        return wrapper
    
    @classmethod
    def list_styles(cls):
        return list(cls._registry.keys())


@OneEarthAttention.register('pangu')
class PanguEarthAttention(nn.Module):
    """
    Pangu-Weather风格的Earth Attention实现。
    
    具有地球位置偏置的窗口注意力机制，支持2D/3D数据。
    自动根据input_resolution维度判断处理模式。
    
    Args:
        dim (int): 输入通道数
        input_resolution (tuple): 输入空间分辨率
            - 2D: (lat, lon)
            - 3D: (pressure_levels, lat, lon)
        window_size (tuple): 窗口大小，形状需与input_resolution匹配
            - 2D: (Wlat, Wlon)
            - 3D: (Wpl, Wlat, Wlon)
        num_heads (int): 注意力头数量
        qkv_bias (bool, optional): 是否在QKV上添加偏置，默认为True
        qk_scale (float, optional): 覆盖默认的QK缩放系数 (head_dim ** -0.5)，默认为None
        attn_drop (float, optional): 注意力权重的dropout比例，默认为0.0
        proj_drop (float, optional): 输出的dropout比例，默认为0.0
        **kwargs: 额外参数（pangu风格不使用）
    
    
    形状:
        - 2D输入: (B×num_lon, num_lat, N, C) -> (B×num_lon, num_lat, N, C)
          其中 N = Wlat × Wlon
        - 3D输入: (B×num_lon, num_pl×num_lat, N, C) -> (B×num_lon, num_pl×num_lat, N, C)
          其中 N = Wpl × Wlat × Wlon
    
    Examples:
        >>> # 2D地表变量注意力
        >>> attn = OneEarthAttention(
        ...     dim=128,
        ...     input_resolution=(128, 256),
        ...     window_size=(8, 8),
        ...     num_heads=4,
        ...     style='pangu'
        ... )
        >>> # num_lat = 128 // 8 = 16
        >>> # num_lon = 256 // 8 = 32
        >>> # 输入: (B*num_lon, num_lat, Wlat*Wlon, C)
        >>> #      = (4*32, 16, 8*8, 128)
        >>> #      = (128, 16, 64, 128)
        >>> x = torch.randn(128, 16, 64, 128)
        >>> out = attn(x)
        >>> out.shape
        torch.Size([128, 16, 64, 128])
        
        >>> # 3D大气变量注意力
        >>> attn = OneEarthAttention(
        ...     dim=192,
        ...     input_resolution=(13, 128, 256),
        ...     window_size=(1, 8, 8),
        ...     num_heads=6,
        ...     style='pangu'
        ... )
        >>> # num_pl = 13 // 1 = 13
        >>> # num_lat = 128 // 8 = 16
        >>> # num_lon = 256 // 8 = 32
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
        **kwargs
    ):
        super().__init__()
        
        if kwargs:
            warnings.warn(
                f"PanguEarthAttention received unexpected kwargs: {list(kwargs.keys())}. "
                f"These will be ignored.",
                UserWarning
            )
        
        if len(input_resolution) != len(window_size):
            raise ValueError(
                f"input_resolution and window_size dimension mismatch: "
                f"input_resolution has {len(input_resolution)} dimensions, "
                f"but window_size has {len(window_size)} dimensions"
            )
        
        if len(input_resolution) not in [2, 3]:
            raise ValueError(
                f"Only support 2D or 3D input, got {len(input_resolution)}D"
            )
        
        self.ndim = len(input_resolution)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        
        if self.ndim == 2:
            self.type_of_windows = input_resolution[0] // window_size[0]
            bias_table_size = (window_size[0] ** 2) * (window_size[1] * 2 - 1)
        else:
            self.type_of_windows = (input_resolution[0] // window_size[0]) * \
                                   (input_resolution[1] // window_size[1])
            bias_table_size = (window_size[0] ** 2) * \
                             (window_size[1] ** 2) * \
                             (window_size[2] * 2 - 1)
        
        self.earth_position_bias_table = nn.Parameter(
            torch.zeros(bias_table_size, self.type_of_windows, num_heads)
        )
        
        earth_position_index = get_earth_position_index(
            window_size,
            self.ndim
        )
        self.register_buffer("earth_position_index", earth_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.earth_position_bias_table = trunc_normal_(
            self.earth_position_bias_table, std=0.02
        )
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        B_, nW_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, nW_, N, 3, self.num_heads, C // self.num_heads)
            .permute(3, 0, 4, 1, 2, 5)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        
        window_elements = 1
        for w in self.window_size:
            window_elements *= w
        
        earth_position_bias = self.earth_position_bias_table[
            self.earth_position_index.view(-1)
        ].view(
            window_elements,
            window_elements,
            self.type_of_windows,
            -1,
        )
        earth_position_bias = earth_position_bias.permute(
            3, 2, 0, 1
        ).contiguous()
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