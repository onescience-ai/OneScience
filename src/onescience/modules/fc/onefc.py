import torch.nn as nn
from torch import Tensor
from .fclayer import ConvNdFCLayer

class OneFc(nn.torch):
    def __init__(self,style="ConvNdFCLayer"):
        if style=="ConvNdFCLaye":
            self.ConvNdFCLayer=ConvNdFCLayer()
        else:
            raise NotImplementedError