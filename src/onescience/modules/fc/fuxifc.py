import torch.nn as nn


class FuXiFC(nn.Module):
    def __init__(
        self, in_channels=1536, out_channels=70*4*4,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fc = nn.Linear(self.in_channels, self.out_channels)
    def forward(self, x):
        return self.fc(x)
