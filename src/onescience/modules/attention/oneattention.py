import torch
from torch import nn
from .earthattention2d import EarthAttention2D

class OneAttention(nn.Module):
    def __init__(self, style="Earth"):
        
        if style == "Earth":
            self.EarthAttention2D = EarthAttention2D( dim, input_resolution, window_size, num_heads,)
        else:
            raise NotImplementedError
