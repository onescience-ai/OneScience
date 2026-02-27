import torch
from torch import nn
import warnings


class OnePatchEmbed:
    """
    统一的Patch嵌入接口，支持多种实现风格。
    
    将图像分割为不重叠的patch并嵌入到向量空间，支持2D/3D数据。
    
    Args:
        img_size (tuple): 输入图像尺寸
            - 2D: (H, W)
            - 3D: (P, H, W)
        patch_size (tuple): 每个patch的大小，形状需与img_size匹配
            - 2D: (patch_h, patch_w)
            - 3D: (patch_p, patch_h, patch_w)
        in_chans (int): 输入图像通道数
        embed_dim (int): 每个patch嵌入后的向量维度
        norm_layer (nn.Module, optional): 归一化层，默认为None。常用: nn.LayerNorm
        style (str): Patch嵌入实现风格，默认'pangu'
            可选值: ['pangu']
        **kwargs: 各style特定参数
    
    Examples:
        >>> # 2D图像 patch嵌入
        >>> patch_embed = OnePatchEmbed(
        ...     img_size=(128, 256),
        ...     patch_size=(4, 4),
        ...     in_chans=3,
        ...     embed_dim=96,
        ...     style='pangu'
        ... )
        
        >>> # 3D图像 patch嵌入
        >>> patch_embed = OnePatchEmbed(
        ...     img_size=(13, 128, 256),
        ...     patch_size=(1, 4, 4),
        ...     in_chans=5,
        ...     embed_dim=192,
        ...     style='pangu'
        ... )
    """
    
    _registry = {}
    
    def __new__(cls, img_size, patch_size, in_chans, embed_dim, 
                norm_layer=None, style='pangu', **kwargs):
        if style not in cls._registry:
            available_styles = list(cls._registry.keys())
            warnings.warn(
                f"Style '{style}' not available. Available styles: {available_styles}. "
                f"Using 'pangu' as fallback.",
                UserWarning
            )
            style = 'pangu'
        
        return cls._registry[style](
            img_size, patch_size, in_chans, embed_dim, norm_layer, **kwargs
        )
    
    @classmethod
    def register(cls, name):
        def wrapper(patch_embed_class):
            cls._registry[name] = patch_embed_class
            return patch_embed_class
        return wrapper
    
    @classmethod
    def list_styles(cls):
        return list(cls._registry.keys())


@OnePatchEmbed.register('pangu')
class PanguPatchEmbed(nn.Module):
    """
    Pangu-Weather风格的Patch嵌入实现。
    
    将图像分割为不重叠的patch并嵌入到向量空间，支持2D/3D数据。
    自动根据img_size维度判断处理模式。
    
    Args:
        img_size (tuple): 输入图像尺寸
            - 2D: (H, W)
            - 3D: (P, H, W)
        patch_size (tuple): 每个patch的大小，形状需与img_size匹配
            - 2D: (patch_h, patch_w)
            - 3D: (patch_p, patch_h, patch_w)
        in_chans (int): 输入图像通道数
        embed_dim (int): 每个patch嵌入后的向量维度

    
    形状:
        - 2D输入: (B, C, H, W) -> (B, embed_dim, H', W')
          其中 H' = ⌈H / patch_h⌉, W' = ⌈W / patch_w⌉
        - 3D输入: (B, C, P, H, W) -> (B, embed_dim, P', H', W')
          其中 P' = ⌈P / patch_p⌉, H' = ⌈H / patch_h⌉, W' = ⌈W / patch_w⌉
    
    Examples:
        >>> # 2D patch嵌入
        >>> patch_embed = OnePatchEmbed(
        ...     img_size=(128, 256),
        ...     patch_size=(4, 4),
        ...     in_chans=3,
        ...     embed_dim=96,
        ...     style='pangu'
        ... )
        >>> x = torch.randn(8, 3, 128, 256)
        >>> out = patch_embed(x)
        >>> out.shape
        torch.Size([8, 96, 32, 64])
        
        >>> # 3D patch嵌入
        >>> patch_embed = OnePatchEmbed(
        ...     img_size=(13, 128, 256),
        ...     patch_size=(1, 4, 4),
        ...     in_chans=5,
        ...     embed_dim=192,
        ...     style='pangu'
        ... )
        >>> x = torch.randn(4, 5, 13, 128, 256)
        >>> out = patch_embed(x)
        >>> out.shape
        torch.Size([4, 192, 13, 32, 64])
        
        >>> # 使用LayerNorm
        >>> patch_embed = OnePatchEmbed(
        ...     img_size=(128, 256),
        ...     patch_size=(4, 4),
        ...     in_chans=3,
        ...     embed_dim=96,
        ...     norm_layer=nn.LayerNorm,
        ...     style='pangu'
        ... )
    """
    
    def __init__(self, img_size, patch_size, in_chans, embed_dim, norm_layer=None, **kwargs):
        super().__init__()
        
        if kwargs:
            warnings.warn(
                f"PanguPatchEmbed received unexpected kwargs: {list(kwargs.keys())}. "
                f"These will be ignored.",
                UserWarning
            )
        
        if len(img_size) != len(patch_size):
            raise ValueError(
                f"img_size and patch_size dimension mismatch: "
                f"img_size has {len(img_size)} dimensions, "
                f"but patch_size has {len(patch_size)} dimensions"
            )
        
        if len(img_size) not in [2, 3]:
            raise ValueError(
                f"Only support 2D or 3D images, got {len(img_size)}D"
            )
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.ndim = len(img_size)
        
        padding = self._compute_padding(img_size, patch_size)
        
        if self.ndim == 2:
            self.pad = nn.ZeroPad2d(padding)
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        else:
            self.pad = nn.ZeroPad3d(padding)
            self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None
    
    def _compute_padding(self, img_size, patch_size):
        paddings = []
        for size, p_size in zip(reversed(img_size), reversed(patch_size)):
            remainder = size % p_size
            if remainder:
                pad_total = p_size - remainder
                pad_start = pad_total // 2
                pad_end = pad_total - pad_start
            else:
                pad_start = pad_end = 0
            paddings.extend([pad_start, pad_end])
        
        return tuple(paddings)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pad(x)
        x = self.proj(x)
        
        if self.norm is not None:
            if self.ndim == 2:
                x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            else:
                x = self.norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        
        return x