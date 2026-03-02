# from .timestepembedder import TimestepEmbedder 
from torch import nn
from .fengwudecoder import FengWuDecoder

_DECODER_REGISTRY = {
    "FengWuDecoder": FengWuDecoder,
}

class OneDecoder(nn.Module):
   
    def __init__(self, style: str, **kwargs):
        super().__init__()

        if style not in _DECODER_REGISTRY:
            raise NotImplementedError(f"Unknown style: {style}")
        
        self.decoder = _DECODER_REGISTRY[style](**kwargs)

    def forward(self, x):
        
        return self.decoder(x) 