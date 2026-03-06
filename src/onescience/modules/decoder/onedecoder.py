# from .timestepembedder import TimestepEmbedder 
from torch import nn
from .fengwudecoder import FengWuDecoder
# from .graphvit_decoder import GraphViTDecoder
from .mesh_graph_decoder import MeshGraphDecoder

_DECODER_REGISTRY = {
    "FengWuDecoder": FengWuDecoder,
    # "GraphViTDecoder": GraphViTDecoder,
    "MeshGraphDecoder": MeshGraphDecoder,
}

class OneDecoder(nn.Module):
   
    def __init__(self, style: str, **kwargs):
        super().__init__()

        if style not in _DECODER_REGISTRY:
            raise NotImplementedError(f"Unknown style: {style}")
        
        self.decoder = _DECODER_REGISTRY[style](**kwargs)

    def forward(self, x):
        
        return self.decoder(x) 