import torch

from torch import nn

from .xihetransformer import XiHeTransformer3D 
from .fuxitransformer import FuXiTransformer

_EMBEDDER_REGISTRY = {
    "XiHeTransformer3D": XiHeTransformer3D,
    "FuXiTransformer": FuXiTransformer

}

class OneTransformer(nn.Module):
    def __init__(self, style="XiHeTransformer3D", **kwargs):
        
        super().__init__()

        if style not in _EMBEDDER_REGISTRY:
            raise NotImplementedError(f"Unknown style: {style}")
        
        self.style = style
        self.transformer = _EMBEDDER_REGISTRY[style](**kwargs)


    def forward(self, x):
        
        return self.transformer(x) 
      