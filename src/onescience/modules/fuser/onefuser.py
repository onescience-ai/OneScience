import torch
from torch import nn

from pangufuser import PanGuFuser

class Onefuser(nn.Module):
   
    def __inin__(self, *args, **kwargs):
        super().__init__()

        if "style" not in kwargs:
           raise ValueError("style must be specified")
        
        style = kwargs.pop("style")

        if style not in ["PanGuFuser"]:
         raise ValueError(f"Unknown style: {style}")
        
        if style == "PanGuFuser":
           self.Fuser = PanGuFuser(args, kwargs)
        
        else:
           raise NotImplementedError
        
    def forward(self, x):
        
        return self.Fuser(x) 