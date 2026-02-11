import torch
from torch import nn
from .pangudownsample2d import PanGuDownSample2D

from .fuxidownsample import FuXiDownSample 
from .fuxiupsample import FuXiUpSample

_EMBEDDER_REGISTRY = {
    "FuXiDownSample": FuXiDownSample,
    "FuXiUpSample": FuXiUpSample,
}

class OneSample(nn.Module):
   
    def __init__(self, style="PanGuDownSample2D", **kwargs):
        super().__init__()

        if style not in _EMBEDDER_REGISTRY:
            raise NotImplementedError(f"Unknown style: {style}")

        self.style = style
        self.sample = _EMBEDDER_REGISTRY[style](**kwargs)

    def forward(self, x):
        
        return self.sample(x) 
 