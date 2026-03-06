import torch
from torch import nn

from .fno_layers import SpectralConv1d as FNOSpectralConv1d
from .fno_layers import SpectralConv2d as FNOSpectralConv2d
from .fno_layers import SpectralConv3d as FNOSpectralConv3d

from .ffno_layers import SpectralConv1d as FFNOSpectralConv1d
from .ffno_layers import SpectralConv2d as FFNOSpectralConv2d
from .ffno_layers import SpectralConv3d as FFNOSpectralConv3d

from .geo_spectral import GeoSpectralConv2d, GeoSpectralConv3d
from .group_spectral import GSpectralConv2d, GSpectralConv3d

from .WaveletFourierKernel import WaveletFourierKernel1D, WaveletFourierKernel2D, WaveletFourierKernel3D
from .WaveletSpatialKernel import WaveletSpatialKernel2D, WaveletSpatialKernel3D
from .MultiWaveletTransform import MultiWaveletTransform1D, MultiWaveletTransform2D, MultiWaveletTransform3D
# 构建统一的注册表
_FOURIER_REGISTRY = {
    "FNOSpectralConv1d": FNOSpectralConv1d,
    "FNOSpectralConv2d": FNOSpectralConv2d,
    "FNOSpectralConv3d": FNOSpectralConv3d,
    "FFNOSpectralConv1d": FFNOSpectralConv1d,
    "FFNOSpectralConv2d": FFNOSpectralConv2d,
    "FFNOSpectralConv3d": FFNOSpectralConv3d,
    "GeoSpectralConv2d": GeoSpectralConv2d,
    "GeoSpectralConv3d": GeoSpectralConv3d,
    "GSpectralConv2d": GSpectralConv2d,
    "GSpectralConv3d": GSpectralConv3d,
    "WaveletFourierKernel1D": WaveletFourierKernel1D,
    "WaveletFourierKernel2D": WaveletFourierKernel2D,
    "WaveletFourierKernel3D": WaveletFourierKernel3D,
    "WaveletSpatialKernel2D": WaveletSpatialKernel2D,
    "WaveletSpatialKernel3D": WaveletSpatialKernel3D,
    "MultiWaveletTransform1D": MultiWaveletTransform1D,
    "MultiWaveletTransform2D": MultiWaveletTransform2D,
    "MultiWaveletTransform3D": MultiWaveletTransform3D,
}

class OneFourier(nn.Module):
    def __init__(self, style: str, **kwargs):
        super().__init__()

        if style not in _FOURIER_REGISTRY:
            raise NotImplementedError(
                f"Unknown style: '{style}'. Available options are: {list(_FOURIER_REGISTRY.keys())}"
            )
        
        # 实例化具体的傅里叶层
        self.fourier_layer = _FOURIER_REGISTRY[style](**kwargs)

    def forward(self, *args, **kwargs):
        """
        前向传播。
        """
        return self.fourier_layer(*args, **kwargs)