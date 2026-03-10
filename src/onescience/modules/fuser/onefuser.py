from torch import nn

from .pangufuser import PanguFuser
from .pangudistributedfuser import PanguDistributedFuser
from .fengwufuser import FengWuFuser
from .fourcastnetfuser import FourCastNetFuser
from .xihelocalsiefuser import XiheLocalSIEFuser
from .xiheglobalsiefuser import XiheGlobalSIEFuser
from .xihefuse import XiheFuser

_FUSER_REGISTRY = {
    "PanguFuser": PanguFuser,
    "PanguDistributedFuser": PanguDistributedFuser,
    "FengWuFuser": FengWuFuser,
    "FourCastNetFuser": FourCastNetFuser,
    "XiheLocalSIEFuser":XiheLocalSIEFuser,
    "XiheGlobalSIEFuser":XiheGlobalSIEFuser,
    "XiheFuser":XiheFuser,
}

class OneFuser(nn.Module):
    """OneFuser module for fusing operations."""
   
    def __init__(self, style: str, **kwargs):
        super().__init__()

        if style not in _FUSER_REGISTRY:
            raise NotImplementedError(f"Unknown style: {style}")
        
        self.Fuser = _FUSER_REGISTRY[style](**kwargs)
         
    def forward(self, x):
        
        return self.Fuser(x) 


      