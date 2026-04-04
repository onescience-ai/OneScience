import torch
from torch import nn

from .pangupatchrecovery import PanguPatchRecovery
from .pangupatchrecovery2d import PanguPatchRecovery2D
from .pangupatchrecovery3d import PanguPatchRecovery3D
from .xihepatchrecovery    import XihePatchRecovery

_RECOVERY_REGISTRY = {
    "PanguPatchRecovery": PanguPatchRecovery,
    "PanguPatchRecovery3D": PanguPatchRecovery3D,
    "PanguPatchRecovery2D": PanguPatchRecovery2D,
    "XihePatchRecovery":XihePatchRecovery,
}

class OneRecovery(nn.Module):
    def __init__(self, style: str, **kwargs):
        super().__init__()

        if style not in _RECOVERY_REGISTRY:
            raise NotImplementedError(f"Unknown style: {style}")
        
        self.Reconvery = _RECOVERY_REGISTRY[style](**kwargs)
        
    def forward(self, x):
        
        return self.Reconvery(x) 
