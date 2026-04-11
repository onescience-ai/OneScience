from torch import nn

from .unet_decoder import UNetDecoder1D, UNetDecoder2D, UNetDecoder3D
from .graphvit_decoder import GraphViTDecoder
from .mesh_graph_decoder import MeshGraphDecoder
from .fengwudecoder import FengWuDecoder
from .protenixdecoder import ProtenixAtomAttentionDecoder

_DECODER_REGISTRY = {
    "UNetDecoder1D": UNetDecoder1D,
    "UNetDecoder2D": UNetDecoder2D,
    "UNetDecoder3D": UNetDecoder3D,
    "GraphViTDecoder": GraphViTDecoder,
    "MeshGraphDecoder": MeshGraphDecoder,
    "FengWuDecoder": FengWuDecoder,
    "ProtenixAtomAttentionDecoder": ProtenixAtomAttentionDecoder,
}

class OneDecoder(nn.Module):
    """
    Decoder 统一入口。

    通过 `style` 从注册表中选择具体解码器实现。
    当前天气相关模型中，常用实现包括：

    - `FengWuDecoder`
    """

    def __init__(self, style: str, **kwargs):
        super().__init__()

        if style not in _DECODER_REGISTRY:
            raise NotImplementedError(
                f"Unknown style: '{style}'. Available options are: {list(_DECODER_REGISTRY.keys())}"
            )

        self.decoder = _DECODER_REGISTRY[style](**kwargs)

    def forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.decoder, name)
