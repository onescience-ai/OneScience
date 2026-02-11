from torch import nn

from .pangufuser import PanGuFuser

_FUSER_REGISTRY = {
    "PanGuFuser": PanGuFuser,
}

class OneFuser(nn.Module):
    """OneFuser module for fusing operations."""
   
    def __init__(self, style: str, **kwargs):
        super().__init__()

        if style not in _FUSER_REGISTRY:
            raise NotImplementedError(f"Unknown style: {style}")
        
        self.Fuser = _FUSER_REGISTRY[style](**kwargs)
         
    def forward(self, x):
        
        return self.Fuser(x) 


      