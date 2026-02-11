import torch

from torch import nn

from .xihetransformer import EarthAttention3D

class OneTransformer(nn.module):
    def __init__(self, style="XiHeTransformer3D"):
        
        if self.style == "XiHeTransformer3D":
            self.XiHeTransformer3D = XiHeTransformer3D(self, dim, input_resolution, window_size, num_heads,)
        else:
            raise NotImplementedError
      