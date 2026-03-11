from torch import nn
from .fengwudecoder import FengWuDecoder
from .mesh_graph_decoder import MeshGraphDecoder
from .protenixdecoder import ProtenixAtomAttentionDecoder

_DECODER_REGISTRY = {
    "FengWuDecoder": FengWuDecoder,
    "MeshGraphDecoder": MeshGraphDecoder,
    "ProtenixAtomAttentionDecoder": ProtenixAtomAttentionDecoder,
}

class OneDecoder(nn.Module):
    def __init__(self, style: str, **kwargs):
        super().__init__()

        if style not in _DECODER_REGISTRY:
            raise NotImplementedError(f"Unknown style: {style}")

        self.decoder = _DECODER_REGISTRY[style](**kwargs)

    def forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.decoder, name)

