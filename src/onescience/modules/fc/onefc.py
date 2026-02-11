import torch.nn as nn
from torch import Tensor
# from .fclayer import ConvNdFCLayer
from .fuxifc import FuXiFC

_EMBEDDER_REGISTRY = {
    "FuXiFC": FuXiFC,

}


class OneFc(nn.Module):
   
    def __init__(self, style="ConvNdFCLayer", **kwargs):
        
        super().__init__()

        if style not in _EMBEDDER_REGISTRY:
            raise NotImplementedError(f"Unknown style: {style}")
        
        self.style = style
        self.fc = _EMBEDDER_REGISTRY[style](**kwargs)
    
    def forward(self, x):
        
        return self.fc(x) 