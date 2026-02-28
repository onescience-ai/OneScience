# from .timestepembedder import TimestepEmbedder 
from torch import nn

from .fuxitransformer import FuxiTransformer

_TRANSFORMER_REGISTRY = {
    "FuxiTransformer": FuxiTransformer,
}

class OneTransformer(nn.Module):
   
    def __init__(self, style: str, **kwargs):
        super().__init__()

        if style not in _TRANSFORMER_REGISTRY:
            raise NotImplementedError(f"Unknown style: {style}")
        
        self.transformer = _TRANSFORMER_REGISTRY[style](**kwargs)

    def forward(self, x):
        
        return self.transformer(x) 