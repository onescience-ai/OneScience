from torch import nn

from onescience.modules._lazy import instantiate_registered_style

_DECODER_REGISTRY = {
    "UNetDecoder1D": ("onescience.modules.decoder.unet_decoder", "UNetDecoder1D"),
    "UNetDecoder2D": ("onescience.modules.decoder.unet_decoder", "UNetDecoder2D"),
    "UNetDecoder3D": ("onescience.modules.decoder.unet_decoder", "UNetDecoder3D"),
    "GraphViTDecoder": ("onescience.modules.decoder.graphvit_decoder", "GraphViTDecoder"),
    "MeshGraphDecoder": ("onescience.modules.decoder.mesh_graph_decoder", "MeshGraphDecoder"),
    "FengWuDecoder": ("onescience.modules.decoder.fengwudecoder", "FengWuDecoder"),
    "ProtenixAtomAttentionDecoder": (
        "onescience.modules.decoder.protenixdecoder",
        "ProtenixAtomAttentionDecoder",
    ),
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

        self.decoder = instantiate_registered_style(style, _DECODER_REGISTRY, "decoder", **kwargs)

    def forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.decoder, name)
