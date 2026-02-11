import torch
from torch import nn
from .sample import PanGuDownSample2D

class OneSample(nn.module):
   
    def __init__(self, style="PanGuDownSample2D"):
        if style == "PanGuDownSample2D":
            self.PaGuDownSample2D = PanGuDownSample2D()
        else:
            raise NotImplementedError("{} is not implemented".format(style))
 