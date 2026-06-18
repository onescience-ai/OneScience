from torch import nn

from onescience.modules._lazy import instantiate_registered_style

# 构建统一的注册表
_FOURIER_REGISTRY = {
    "FNOSpectralConv1d": ("onescience.modules.fourier.fno_layers", "SpectralConv1d"),
    "FNOSpectralConv2d": ("onescience.modules.fourier.fno_layers", "SpectralConv2d"),
    "FNOSpectralConv3d": ("onescience.modules.fourier.fno_layers", "SpectralConv3d"),
    "FFNOSpectralConv1d": ("onescience.modules.fourier.ffno_layers", "SpectralConv1d"),
    "FFNOSpectralConv2d": ("onescience.modules.fourier.ffno_layers", "SpectralConv2d"),
    "FFNOSpectralConv3d": ("onescience.modules.fourier.ffno_layers", "SpectralConv3d"),
    "GeoSpectralConv2d": ("onescience.modules.fourier.geo_spectral", "GeoSpectralConv2d"),
    "GeoSpectralConv3d": ("onescience.modules.fourier.geo_spectral", "GeoSpectralConv3d"),
    "GSpectralConv2d": ("onescience.modules.fourier.group_spectral", "GSpectralConv2d"),
    "GSpectralConv3d": ("onescience.modules.fourier.group_spectral", "GSpectralConv3d"),
    "WaveletFourierKernel1D": (
        "onescience.modules.fourier.WaveletFourierKernel",
        "WaveletFourierKernel1D",
    ),
    "WaveletFourierKernel2D": (
        "onescience.modules.fourier.WaveletFourierKernel",
        "WaveletFourierKernel2D",
    ),
    "WaveletFourierKernel3D": (
        "onescience.modules.fourier.WaveletFourierKernel",
        "WaveletFourierKernel3D",
    ),
    "WaveletSpatialKernel2D": (
        "onescience.modules.fourier.WaveletSpatialKernel",
        "WaveletSpatialKernel2D",
    ),
    "WaveletSpatialKernel3D": (
        "onescience.modules.fourier.WaveletSpatialKernel",
        "WaveletSpatialKernel3D",
    ),
    "MultiWaveletTransform1D": (
        "onescience.modules.fourier.MultiWaveletTransform",
        "MultiWaveletTransform1D",
    ),
    "MultiWaveletTransform2D": (
        "onescience.modules.fourier.MultiWaveletTransform",
        "MultiWaveletTransform2D",
    ),
    "MultiWaveletTransform3D": (
        "onescience.modules.fourier.MultiWaveletTransform",
        "MultiWaveletTransform3D",
    ),
}

class OneFourier(nn.Module):
    def __init__(self, style: str, **kwargs):
        super().__init__()

        self.fourier_layer = instantiate_registered_style(
            style,
            _FOURIER_REGISTRY,
            "fourier",
            **kwargs,
        )

    def forward(self, *args, **kwargs):
        """
        前向传播。
        """
        return self.fourier_layer(*args, **kwargs)
