from torch import nn

from .pangudownsample2d import PanguDownSample2D
from .panguupsample2d import PanguUpSample2D
from .pangudownsample3d import PanguDownSample3D
from .panguupsample3d import PanguUpSample3D

_SAMPLER_REGISTRY = {
    "PanguDownSample2D": PanguDownSample2D,
    "PanguDownSample3D": PanguDownSample3D,
    "PanguUpSample2D": PanguUpSample2D,
    "PanguUpSample3D": PanguUpSample3D,
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
