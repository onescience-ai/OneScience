from torch import nn

from .earthattention2d import EarthAttention2D
from .earthattention3d import EarthAttention3D
from .xihefeaturegroupattention import FeatureGroupingAttention
from .xihefeatureungroupattention import FeatureUngroupingAttention


_ATTENTIONER_REGISTRY = {
    "EarthAttention2D": EarthAttention2D,
    "EarthAttention3D": EarthAttention3D,
    "FeatureUngroupingAttention":FeatureUngroupingAttention,
    "FeatureGroupingAttention":FeatureGroupingAttention
    
}

class OneAttention(nn.Module):
    def __init__(self, style: str, **kwargs):
        super().__init__()

        if style not in _ATTENTIONER_REGISTRY:
            raise NotImplementedError(f"Unknown style: {style}")
        
        self.attentioner = _ATTENTIONER_REGISTRY[style](**kwargs)

    def forward(self, x, mask=None):
        return self.attentioner(x, mask) 

    