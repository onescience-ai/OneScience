import torch
from torch import nn
from torch.nn import functional as F
from typing import Sequence


class FuxiFC(nn.Module):

    def __init__(self,
                 in_channels=1536,
                 out_channels=70*4*4):
        super().__init__()
        
        self.fc = nn.Linear(in_channels, out_channels)
    def forward(self, x: torch.Tensor):
        x = self.fc(x)  # B Lat Lon C
        return x