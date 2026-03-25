from torch import nn

from .earthattention2d import EarthAttention2D
from .earthattention3d import EarthAttention3D
from .physicsattention import Physics_Attention_Irregular_Mesh
from .physicsattention import Physics_Attention_Irregular_Mesh_plus
from .physicsattention import Physics_Attention_Structured_Mesh_1D
from .physicsattention import Physics_Attention_Structured_Mesh_2D
from .physicsattention import Physics_Attention_Structured_Mesh_3D
from .factattention import FactAttention2D
from .factattention import FactAttention3D
from .flashattention import FlashAttention
from .linearattention import LinearAttention
from .linearattention import Vanilla_Linear_Attention
from .multiheadattention import MultiHeadAttention
from .selfattention import SelfAttention
from .windowattention import WindowAttention
from .nystrom_attention import NystromAttention
from .xihefeaturegroupattention import FeatureGroupingAttention
from .xihefeatureungroupattention import FeatureUngroupingAttention
from .protenixattention import (
    ProtenixAttention,
    ProtenixAttentionPairBias,
    ProtenixAttentionPairBiasWithLocalAttn,
)
_ATTENTIONER_REGISTRY = {
    "EarthAttention2D": EarthAttention2D,
    "EarthAttention3D": EarthAttention3D,
    "Physics_Attention_Irregular_Mesh": Physics_Attention_Irregular_Mesh,
    "Physics_Attention_Irregular_Mesh_plus": Physics_Attention_Irregular_Mesh_plus,
    "Physics_Attention_Structured_Mesh_1D": Physics_Attention_Structured_Mesh_1D,
    "Physics_Attention_Structured_Mesh_2D": Physics_Attention_Structured_Mesh_2D,
    "Physics_Attention_Structured_Mesh_3D": Physics_Attention_Structured_Mesh_3D,
    "FactAttention2D": FactAttention2D,
    "FactAttention3D": FactAttention3D,
    "FlashAttention": FlashAttention,
    "LinearAttention": LinearAttention,
    "Vanilla_Linear_Attention": Vanilla_Linear_Attention,
    "MultiHeadAttention": MultiHeadAttention,
    "SelfAttention": SelfAttention,
    "WindowAttention": WindowAttention,
    "NystromAttention": NystromAttention,
    "FeatureUngroupingAttention": FeatureUngroupingAttention,
    "FeatureGroupingAttention": FeatureGroupingAttention,
    "ProtenixAttention": ProtenixAttention,
    "ProtenixAttentionPairBias": ProtenixAttentionPairBias,
    "ProtenixAttentionPairBiasWithLocalAttn": ProtenixAttentionPairBiasWithLocalAttn,
}


class OneAttention(nn.Module):
    def __init__(self, style: str, **kwargs):
        super().__init__()

        if style not in _ATTENTIONER_REGISTRY:
            raise NotImplementedError(f"Unknown style: {style}")

        self.attentioner = _ATTENTIONER_REGISTRY[style](**kwargs)

    def forward(self, *args, **kwargs):
        return self.attentioner(*args, **kwargs)