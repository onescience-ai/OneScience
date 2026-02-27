import torch
from torch import nn
import warnings


class OnePatchRecovery:
    """
    统一的Patch恢复接口，支持多种实现风格。
    
    将patch嵌入恢复为原始图像尺寸，支持2D/3D数据。
    
    Args:
        img_size (tuple): 目标图像尺寸
            - 2D: (H, W)
            - 3D: (P, H, W)
        patch_size (tuple): 每个patch的大小，形状需与img_size匹配
            - 2D: (patch_h, patch_w)
            - 3D: (patch_p, patch_h, patch_w)
        in_chans (int): 输入特征通道数
        out_chans (int): 输出图像通道数
        style (str): Patch恢复实现风格，默认'pangu'
            可选值: ['pangu']
        **kwargs: 各style特定参数（当前pangu不需要额外参数）
    
    Examples:
        >>> # 2D图像恢复
        >>> patch_recovery = OnePatchRecovery(
        ...     img_size=(128, 256),
        ...     patch_size=(4, 4),
        ...     in_chans=96,
        ...     out_chans=3,
        ...     style='pangu'
        ... )
        
        >>> # 3D图像恢复
        >>> patch_recovery = OnePatchRecovery(
        ...     img_size=(13, 128, 256),
        ...     patch_size=(1, 4, 4),
        ...     in_chans=192,
        ...     out_chans=5,
        ...     style='pangu'
        ... )
    """
    
    _registry = {}
    
    def __new__(cls, img_size, patch_size, in_chans, out_chans, 
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
            img_size, patch_size, in_chans, out_chans, **kwargs
        )
    
    @classmethod
    def register(cls, name):
        def wrapper(patch_recovery_class):
            cls._registry[name] = patch_recovery_class
            return patch_recovery_class
        return wrapper
    
    @classmethod
    def list_styles(cls):
        return list(cls._registry.keys())


@OnePatchRecovery.register('pangu')
class PanguPatchRecovery(nn.Module):
    """
    Pangu-Weather风格的Patch恢复实现。
    
    将patch嵌入恢复为原始图像尺寸，支持2D/3D数据。
    自动根据img_size维度判断处理模式。
    
    Args:
        img_size (tuple): 目标图像尺寸
            - 2D: (H, W)
            - 3D: (P, H, W)
        patch_size (tuple): 每个patch的大小，形状需与img_size匹配
            - 2D: (patch_h, patch_w)
            - 3D: (patch_p, patch_h, patch_w)
        in_chans (int): 输入特征通道数
        out_chans (int): 输出图像通道数 

    形状:
        - 2D输入: (B, in_chans, H', W') -> (B, out_chans, H, W)
        - 3D输入: (B, in_chans, P', H', W') -> (B, out_chans, P, H, W)
    
    Examples:
        >>> # 2D图像恢复
        >>> patch_recovery = OnePatchRecovery(
        ...     img_size=(128, 256),
        ...     patch_size=(4, 4),
        ...     in_chans=96,
        ...     out_chans=3,
        ...     style='pangu'
        ... )
        >>> x = torch.randn(8, 96, 32, 64)
        >>> out = patch_recovery(x)
        >>> out.shape
        torch.Size([8, 3, 128, 256])
        
        >>> # 3D图像恢复
        >>> patch_recovery = OnePatchRecovery(
        ...     img_size=(13, 128, 256),
        ...     patch_size=(1, 4, 4),
        ...     in_chans=192,
        ...     out_chans=5,
        ...     style='pangu'
        ... )
        >>> x = torch.randn(4, 192, 13, 32, 64)
        >>> out = patch_recovery(x)
        >>> out.shape
        torch.Size([4, 5, 13, 128, 256])
    """
    
    def __init__(self, img_size, patch_size, in_chans, out_chans, **kwargs):
        super().__init__()
        
        if kwargs:
            warnings.warn(
                f"PanguPatchRecovery received unexpected kwargs: {list(kwargs.keys())}. "
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
        
        if self.ndim == 2:
            self.conv = nn.ConvTranspose2d(in_chans, out_chans, patch_size, patch_size)
        else:
            self.conv = nn.ConvTranspose3d(in_chans, out_chans, patch_size, patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.conv(x)
        spatial_dims = output.shape[2:]
        
        slices = [slice(None), slice(None)]
        
        for out_size, target_size in zip(spatial_dims, self.img_size):
            pad = out_size - target_size
            pad_start = pad // 2
            pad_end = out_size - (pad - pad_start)
            slices.append(slice(pad_start, pad_end))
        
        return output[tuple(slices)]