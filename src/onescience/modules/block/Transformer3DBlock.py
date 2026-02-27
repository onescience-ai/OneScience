import torch
from torch import nn
import warnings
from collections.abc import Sequence

from onescience.modules.func_utils.pangu_utils import (
    get_pad3d,
    crop3d,
    window_partition,
    window_reverse,
    get_shift_window_mask
)
from onescience.modules.layer.pangu_layer import DropPath, Mlp
from onescience.modules.attention.EarthAttention import OneEarthAttention


class OneTransformer3DBlock:
    """
    统一的三维Transformer块接口，支持多种实现风格。
    
    带有移位窗口机制的三维Transformer块，用于处理图像类数据。
    
    Args:
        dim (int): 输入特征通道数
        input_resolution (tuple[int, int, int]): 输入空间分辨率 (pressure_levels, lat, lon)
        num_heads (int): 注意力头数量
        window_size (tuple[int, int, int], optional): 窗口大小 (Wpl, Wlat, Wlon)，默认为 (2, 6, 12)
        shift_size (tuple[int, int, int], optional): 窗口移位大小 (shift_pl, shift_lat, shift_lon)，默认为 (1, 3, 6)
        mlp_ratio (float, optional): MLP隐藏层维度与嵌入维度的比例，默认为 4.0
        qkv_bias (bool, optional): 是否在QKV上添加偏置，默认为 True
        qk_scale (float, optional): 覆盖默认的QK缩放系数，默认为 None
        drop (float, optional): Dropout比例，默认为 0.0
        attn_drop (float, optional): 注意力权重的dropout比例，默认为 0.0
        drop_path (float, optional): 随机深度比例，默认为 0.0
        act_layer (nn.Module, optional): 激活函数层，默认为 nn.GELU
        norm_layer (nn.Module, optional): 归一化层，默认为 nn.LayerNorm
        style (str): Transformer块实现风格，默认'pangu'
            可选值: ['pangu']
        **kwargs: 各style特定参数（当前pangu不需要额外参数）
    
    Examples:
        >>> # 基础用法
        >>> block = OneTransformer3DBlock(
        ...     dim=192,
        ...     input_resolution=(13, 128, 256),
        ...     num_heads=6,
        ...     window_size=(2, 6, 12),
        ...     shift_size=(1, 3, 6),
        ...     style='pangu'
        ... )
        
        >>> # 堆叠多层（等价于原FuserLayer）
        >>> blocks = nn.ModuleList([
        ...     OneTransformer3DBlock(
        ...         dim=192,
        ...         input_resolution=(13, 128, 256),
        ...         num_heads=6,
        ...         window_size=(2, 6, 12),
        ...         shift_size=(0, 0, 0) if i % 2 == 0 else (1, 3, 6),
        ...         style='pangu'
        ...     )
        ...     for i in range(6)
        ... ])
    """
    
    _registry = {}
    
    def __new__(cls, dim, input_resolution, num_heads,
                window_size=None, shift_size=None, mlp_ratio=4.0,
                qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0,
                drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
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
            dim, input_resolution, num_heads,
            window_size, shift_size, mlp_ratio,
            qkv_bias, qk_scale, drop, attn_drop,
            drop_path, act_layer, norm_layer, **kwargs
        )
    
    @classmethod
    def register(cls, name):
        def wrapper(transformer_block_class):
            cls._registry[name] = transformer_block_class
            return transformer_block_class
        return wrapper
    
    @classmethod
    def list_styles(cls):
        return list(cls._registry.keys())


@OneTransformer3DBlock.register('pangu')
class PanguTransformer3DBlock(nn.Module):
    """
    Pangu-Weather风格的三维Transformer块实现。
    
    带有移位窗口机制的三维Transformer块，用于处理图像类数据。
    
    Args:
        dim (int): 输入特征通道数
        input_resolution (tuple[int, int, int]): 输入空间分辨率 (pressure_levels, lat, lon)
        num_heads (int): 注意力头数量
        window_size (tuple[int, int, int], optional): 窗口大小 (Wpl, Wlat, Wlon)，默认为 (2, 6, 12)
        shift_size (tuple[int, int, int], optional): 窗口移位大小 (shift_pl, shift_lat, shift_lon)，默认为 (1, 3, 6)
        mlp_ratio (float, optional): MLP隐藏层维度与嵌入维度的比例，默认为 4.0
        qkv_bias (bool, optional): 是否在QKV上添加偏置，默认为 True
        qk_scale (float, optional): 覆盖默认的QK缩放系数，默认为 None
        drop (float, optional): Dropout比例，默认为 0.0
        attn_drop (float, optional): 注意力权重的dropout比例，默认为 0.0
        drop_path (float, optional): 随机深度比例，默认为 0.0
        act_layer (nn.Module, optional): 激活函数层，默认为 nn.GELU
        norm_layer (nn.Module, optional): 归一化层，默认为 nn.LayerNorm
        **kwargs: 额外参数（pangu风格不使用）
    
    形状:
        - 输入: (B, L, C)，其中 L = Pl × Lat × Lon
        - 输出: (B, L, C)
    
    Examples:
        >>> # 基础用法
        >>> # input_resolution=(13, 128, 256), L=13*128*256=425984
        >>> block = OneTransformer3DBlock(
        ...     dim=192,
        ...     input_resolution=(13, 128, 256),
        ...     num_heads=6,
        ...     window_size=(2, 6, 12),
        ...     shift_size=(1, 3, 6),
        ...     style='pangu'
        ... )
        >>> x = torch.randn(4, 425984, 192)
        >>> out = block(x)
        >>> out.shape
        torch.Size([4, 425984, 192])
        
        >>> # 使用默认window_size和shift_size
        >>> block = OneTransformer3DBlock(
        ...     dim=192,
        ...     input_resolution=(13, 128, 256),
        ...     num_heads=6,
        ...     style='pangu'
        ... )
        
        >>> # 偶数层使用shift_size=(0,0,0)，奇数层使用默认shift
        >>> blocks = nn.ModuleList([
        ...     OneTransformer3DBlock(
        ...         dim=192,
        ...         input_resolution=(13, 128, 256),
        ...         num_heads=6,
        ...         shift_size=(0, 0, 0) if i % 2 == 0 else None,
        ...         style='pangu'
        ...     )
        ...     for i in range(6)
        ... ])
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
        **kwargs
    ):
        super().__init__()
        
        if kwargs:
            warnings.warn(
                f"PanguTransformer3DBlock received unexpected kwargs: {list(kwargs.keys())}. "
                f"These will be ignored.",
                UserWarning
            )
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
        
        self.attn = OneEarthAttention(
            dim=dim,
            input_resolution=pad_resolution,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            style='pangu'
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
                x, shifts=(-shift_pl, -shift_lat, -shift_lat), dims=(1, 2, 3)
            )
            x_windows = window_partition(shifted_x, self.window_size)
        else:
            shifted_x = x
            x_windows = window_partition(shifted_x, self.window_size)
        
        win_pl, win_lat, win_lon = self.window_size
        x_windows = x_windows.view(
            x_windows.shape[0], x_windows.shape[1], win_pl * win_lat * win_lon, C
        )
        
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        
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