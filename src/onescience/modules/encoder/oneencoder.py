from torch import nn

from onescience.modules._lazy import instantiate_registered_style

_ENCODER_REGISTRY = {
    "UNetEncoder1D": ("onescience.modules.encoder.unet_encoder", "UNetEncoder1D"),
    "UNetEncoder2D": ("onescience.modules.encoder.unet_encoder", "UNetEncoder2D"),
    "UNetEncoder3D": ("onescience.modules.encoder.unet_encoder", "UNetEncoder3D"),
    "GraphViTEncoder": ("onescience.modules.encoder.graphvit_encoder", "GraphViTEncoder"),
    "MeshGraphEncoder": ("onescience.modules.encoder.mesh_graph_encoder", "MeshGraphEncoder"),
    "FengWuEncoder": ("onescience.modules.encoder.fengwuencoder", "FengWuEncoder"),
    "ProtenixRelativePositionEncoding": (
        "onescience.modules.encoder.protenixencoding",
        "ProtenixRelativePositionEncoding",
    ),
    "ProtenixAtomAttentionEncoder": (
        "onescience.modules.encoder.protenixencoding",
        "ProtenixAtomAttentionEncoder",
    ),
}

class OneEncoder(nn.Module):
    """
    Encoder 统一入口。

    通过 `style` 从注册表中选择具体编码器实现。
    当前天气相关模型中，常用实现包括：

    - `FengWuEncoder`
    """

    def __init__(self, style: str, **kwargs):
        super().__init__()
        self.encoder = instantiate_registered_style(style, _ENCODER_REGISTRY, "encoder", **kwargs)

    def forward(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        new_state = {'encoder.' + k: v for k, v in state_dict.items()}
        return super().load_state_dict(new_state, strict=strict)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.encoder, name)
