import torch
from torch import nn

class PanGuDownSample2D(nn.Module):
    """
    Revise from WeatherLearn https://github.com/lizhuoq/WeatherLearn
    2D Down-sampling operation

    Args:
        in_dim (int): Number of input channels.
        input_resolution (tuple[int]): [latitude, longitude]
        output_resolution (tuple[int]): [latitude, longitude]
    """

    def __init__(self, 
                 input_resolution, 
                 output_resolution,
                 in_dim=192):
        super().__init__()
        
        self.linear = nn.Linear(in_dim * 4, in_dim * 2, bias=False)
        self.norm = nn.LayerNorm(4 * in_dim)
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution

        in_lat, in_lon = self.input_resolution
        out_lat, out_lon = self.output_resolution

        h_pad = out_lat * 2 - in_lat
        w_pad = out_lon * 2 - in_lon

        pad_top = h_pad // 2
        pad_bottom = h_pad - pad_top

        pad_left = w_pad // 2
        pad_right = w_pad - pad_left

        self.pad = nn.ZeroPad2d((pad_left, pad_right, pad_top, pad_bottom))

    def forward(self, x: torch.Tensor):
        B, N, C = x.shape
        in_lat, in_lon = self.input_resolution
        out_lat, out_lon = self.output_resolution
        x = x.reshape(B, in_lat, in_lon, C)

        # Padding the input to facilitate downsampling
        x = self.pad(x.permute(0, -1, 1, 2)).permute(0, 2, 3, 1)
        x = x.reshape(B, out_lat, 2, out_lon, 2, C).permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, out_lat * out_lon, 4 * C)

        x = self.norm(x)
        x = self.linear(x)
        return x
