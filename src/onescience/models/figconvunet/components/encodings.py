import numpy as np
import torch
import torch.nn as nn


class SinusoidalEncoding(nn.Module):
    """SinusoidalEncoding."""

    def __init__(self, num_channels: int, data_range: float = 2.0):
        super().__init__()
        assert num_channels % 2 == 0, (
            f"num_channels must be even for sin/cos, got {num_channels}"
        )
        self.num_channels = num_channels
        self.data_range = data_range

    def forward(self, x):
        freqs = 2 ** torch.arange(
            start=0, end=self.num_channels // 2, device=x.device
        ).to(x.dtype)
        freqs = (2 * np.pi / self.data_range) * freqs
        x = x.unsqueeze(-1)
        # Make freq to have the same dimensions as x. X can be of any shape
        freqs = freqs.reshape((1,) * (len(x.shape) - 1) + freqs.shape)
        x = x * freqs
        x = torch.cat([x.cos(), x.sin()], dim=-1).flatten(start_dim=-2)
        return x
