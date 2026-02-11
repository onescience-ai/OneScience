import torch
from torch import nn

class PanGuUpSample2D(nn.Module):
    """
    改编自 WeatherLearn 项目 https://github.com/lizhuoq/WeatherLearn
    二维上采样操作

    参数:
        in_dim (int): 输入特征的通道数
        out_dim (int): 输出特征的通道数
        input_resolution (tuple[int]): 输入特征的分辨率 [pressure levels, latitude, longitude]
        output_resolution (tuple[int]): 输出特征的分辨率 [pressure levels, latitude, longitude]
    """

    def __init__(self, in_dim, out_dim, input_resolution, output_resolution):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim * 4, bias=False)
        self.linear2 = nn.Linear(out_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): (B, N, C)
        """
        B, N, C = x.shape
        in_lat, in_lon = self.input_resolution
        out_lat, out_lon = self.output_resolution

        x = self.linear1(x)
        x = x.reshape(B, in_lat, in_lon, 2, 2, C // 2).permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, in_lat * 2, in_lon * 2, -1)

        pad_h = in_lat * 2 - out_lat
        pad_w = in_lon * 2 - out_lon

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        x = x[
            :, pad_top : 2 * in_lat - pad_bottom, pad_left : 2 * in_lon - pad_right, :
        ]
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        x = self.norm(x)
        x = self.linear2(x)
        return x