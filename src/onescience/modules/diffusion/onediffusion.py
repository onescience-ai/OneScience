import torch
from torch import nn
from .diffusionmodule import DiffusionModule

class Diffusion(nn.module):
    def __inin__(self, style="DiffusionModule"):
        if style == "DiffusionModule":
            self.DiffusionModule = DiffusionModule()
        else:
            raise ValueError("Unknown diffusion style")