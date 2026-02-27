import torch
from torch import nn


class UpSample3D(nn.Module):
    """
        通过可学习的线性变换和像素重排实现三维空间上采样。

        Args:
            in_dim (int): 输入通道数
            out_dim (int): 输出通道数
            input_resolution (tuple[int, int, int]): 输入空间维度 (P, H, W)
            output_resolution (tuple[int, int, int]): 输出空间维度 (P', H', W'), 其中 P' ≤ P

        形状:
            输入: (B, N, C_in)，其中 N = P × H × W
            输出: (B, M, C_out)，其中 M = P' × H' × W'

        Example:
            >>> upsample = UpSample3D(256, 128, (13, 32, 64), (13, 64, 128))
            >>> x = torch.randn(4, 26624, 256)
            >>> out = upsample(x)
            >>> out.shape
            torch.Size([4, 106496, 128])

    """
    def __init__(self, in_dim, out_dim, input_resolution, output_resolution):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim * 4, bias=False)
        self.linear2 = nn.Linear(out_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution

    def forward(self, x: torch.Tensor):
        B, N, C = x.shape
        in_pl, in_lat, in_lon = self.input_resolution
        out_pl, out_lat, out_lon = self.output_resolution

        x = self.linear1(x)
        x = x.reshape(B, in_pl, in_lat, in_lon, 2, 2, C // 2).permute(
            0, 1, 2, 4, 3, 5, 6
        )
        x = x.reshape(B, in_pl, in_lat * 2, in_lon * 2, -1)

        pad_h = in_lat * 2 - out_lat
        pad_w = in_lon * 2 - out_lon

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        x = x[
            :,
            :out_pl,
            pad_top : 2 * in_lat - pad_bottom,
            pad_left : 2 * in_lon - pad_right,
            :,
        ]
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], x.shape[4])
        x = self.norm(x)
        x = self.linear2(x)
        return x


class UpSample2D(nn.Module):
    """
        通过可学习的线性变换和像素重排实现二维空间上采样。

        Args:
            in_dim (int): 输入通道数
            out_dim (int): 输出通道数
            input_resolution (tuple[int, int]): 输入空间维度 (H, W)
            output_resolution (tuple[int, int]): 输出空间维度 (H', W')

        形状:
            输入: (B, N, C_in)，其中 N = H × W
            输出: (B, M, C_out)，其中 M = H' × W'

        Example:
            >>> upsample = UpSample2D(128, 64, (64, 128), (128, 256))
            >>> x = torch.randn(8, 8192, 128)
            >>> out = upsample(x)
            >>> out.shape
            torch.Size([8, 32768, 64])

    """
    
    def __init__(self, in_dim, out_dim, input_resolution, output_resolution):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim * 4, bias=False)
        self.linear2 = nn.Linear(out_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution

    def forward(self, x: torch.Tensor):

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