import torch
from torch import nn
from .pangupatchrecovery3d import PanGuPatchRecovery3D

class OnePatch(nn.module):
    def __init__(self, style="PanGuPatchRecovery3D"):
        
        if self.style == "PanGuPatchRecovery3D":
            self.PanGuPatchRecovery3D = PanGuPatchRecovery3D(self, dim, input_resolution, window_size, num_heads,)
        else:
            raise NotImplementedError