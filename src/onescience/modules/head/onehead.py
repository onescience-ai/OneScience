import torch 
from torch import nn
from .maskedmsahead import MaskedMSAHead

class OneHead(nn.module):
    def __inin__(self, style="MaskMsAHead"):
        if style == "MaskMsAHead":
            self.MaskMsAHead = MaskedMSAHead()
        else:
            raise NotImplementedError