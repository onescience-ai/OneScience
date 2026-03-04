import torch
from torch import nn

from .fuxifc import FuxiFC
from .fourcastnetfc import FourCastNetFC

_FC_REGISTRY = {
    "FuxiFC": FuxiFC,
    "FourCastNetFC": FourCastNetFC,
}

class OneFC(nn.Module):
    def __init__(self, style: str, **kwargs):
        super().__init__()

        if style not in _FC_REGISTRY:
            raise NotImplementedError(f"Unknown style: {style}")
        
        self.fc = _FC_REGISTRY[style](**kwargs)
        
    def forward(self, x):
        
        return self.fc(x) 