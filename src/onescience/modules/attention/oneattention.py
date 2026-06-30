from torch import nn

from onescience.modules._lazy import instantiate_registered_style

_ATTENTIONER_REGISTRY = {
    "EarthAttention2D": ("onescience.modules.attention.earthattention2d", "EarthAttention2D"),
    "EarthAttention3D": ("onescience.modules.attention.earthattention3d", "EarthAttention3D"),
    "EarthDistributedAttention3D": ("onescience.modules.attention.earthdistributedattention3d", "EarthDistributedAttention3D"),
    "Physics_Attention_Irregular_Mesh": (
        "onescience.modules.attention.physicsattention",
        "Physics_Attention_Irregular_Mesh",
    ),
    "Physics_Attention_Irregular_Mesh_plus": (
        "onescience.modules.attention.physicsattention",
        "Physics_Attention_Irregular_Mesh_plus",
    ),
    "Physics_Attention_Structured_Mesh_1D": (
        "onescience.modules.attention.physicsattention",
        "Physics_Attention_Structured_Mesh_1D",
    ),
    "Physics_Attention_Structured_Mesh_2D": (
        "onescience.modules.attention.physicsattention",
        "Physics_Attention_Structured_Mesh_2D",
    ),
    "Physics_Attention_Structured_Mesh_3D": (
        "onescience.modules.attention.physicsattention",
        "Physics_Attention_Structured_Mesh_3D",
    ),
    "FactAttention2D": ("onescience.modules.attention.factattention", "FactAttention2D"),
    "FactAttention3D": ("onescience.modules.attention.factattention", "FactAttention3D"),
    "FlashAttention": ("onescience.modules.attention.flashattention", "FlashAttention"),
    "LinearAttention": ("onescience.modules.attention.linearattention", "LinearAttention"),
    "Vanilla_Linear_Attention": (
        "onescience.modules.attention.linearattention",
        "Vanilla_Linear_Attention",
    ),
    "MultiHeadAttention": ("onescience.modules.attention.multiheadattention", "MultiHeadAttention"),
    "SelfAttention": ("onescience.modules.attention.selfattention", "SelfAttention"),
    "WindowAttention": ("onescience.modules.attention.windowattention", "WindowAttention"),
    "NystromAttention": ("onescience.modules.attention.nystrom_attention", "NystromAttention"),
    "FeatureUngroupingAttention": (
        "onescience.modules.attention.xihefeatureungroupattention",
        "FeatureUngroupingAttention",
    ),
    "FeatureGroupingAttention": (
        "onescience.modules.attention.xihefeaturegroupattention",
        "FeatureGroupingAttention",
    ),
    "DistributedFeatureUngroupingAttention": (
        "onescience.modules.attention.xihedistributedfeatureungroupattention",
        "DistributedFeatureUngroupingAttention",
    ),
    "DistributedFeatureGroupingAttention": (
        "onescience.modules.attention.xihedistributedfeaturegroupattention",
        "DistributedFeatureGroupingAttention",
    ),
    "ProtenixAttention": ("onescience.modules.attention.protenixattention", "ProtenixAttention"),
    "ProtenixAttentionPairBias": (
        "onescience.modules.attention.protenixattention",
        "ProtenixAttentionPairBias",
    ),
    "ProtenixAttentionPairBiasWithLocalAttn": (
        "onescience.modules.attention.protenixattention",
        "ProtenixAttentionPairBiasWithLocalAttn",
    ),
}


class OneAttention(nn.Module):
    """
    Attention 统一入口。

    通过 `style` 从注册表中选择具体注意力实现。
    当前天气相关模型中，常用实现包括：

    - `EarthAttention2D`
    - `EarthAttention3D`
    """

    def __init__(self, style: str, **kwargs):
        super().__init__()

        self.attentioner = instantiate_registered_style(
            style,
            _ATTENTIONER_REGISTRY,
            "attention",
            **kwargs,
        )

    def forward(self, *args, **kwargs):
        return self.attentioner(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.attentioner, name)