from torch import nn

from .pangudownsample2d import PanGuDownSample2D
from .panguupsample2d import PanGuUpSample2D
from .pangudownsample3d import PanGuDownSample3D
from .panguupsample3d import PanGuUpSample3D

_SAMPLER_REGISTRY = {
    "PanGuDownSample2D": PanGuDownSample2D,
    "PanGuDownSample3D": PanGuDownSample3D,
    "PanGuUpSample2D": PanGuUpSample2D,
    "PanGuUpSample3D": PanGuUpSample3D,
}

class OneSample(nn.Module):
    """OneSample module for sampling operations."""
   
    def __init__(self, style: str, **kwargs):
        super().__init__()

        if style not in _SAMPLER_REGISTRY:
            raise NotImplementedError(f"Unknown style: {style}")
        
        self.Sampler = _SAMPLER_REGISTRY[style](**kwargs)
        
    def forward(self, x):
        
        return self.Sampler(x) 
