import torch
from torch import nn

from .pangupatchrecovery2d import PatchRecovery2D
from .pangupatchrecovery3d import PanGuPatchRecovery3D

_RECOVERY_REGISTRY = {
    "pangupatchrecovery3d": PanGuPatchRecovery3D,
    "pangupatchrecovery2d": PatchRecovery2D,
}

class OneRecovery(nn.Module):
    def __init__(self, style: str, **kwargs):
        super().__init__()

        if style not in _RECOVERY_REGISTRY:
            raise NotImplementedError(f"Unknown style: {style}")
        
        self.Reconvery = _RECOVERY_REGISTRY[style](**kwargs)
        
    def forward(self, x):
        
        return self.Recovery(x) 