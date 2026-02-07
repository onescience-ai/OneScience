import torch

from torch import nn
from .timestepembedder import TimestepEmbedder 
    

class OneEmbedder(nn.Module):
   
   def __inin__(self, style="TimestepEmbedder"):
       if style == "TimestepEmbedder":
           self.embedder = TimestepEmbedder()
       else:
           raise NotImplementedError