import torch
from torch import nn

class DownSample3D(nn.Module):
    """
        通过空间合并和线性变换实现三维空间下采样。

        Args:
            in_dim (int): 输入通道数
            input_resolution (tuple[int, int, int]): 输入空间维度 (P, H, W)
            output_resolution (tuple[int, int, int]): 输出空间维度 (P', H', W')，其中 P' ≤ P

        形状:
            输入: (B, N, C_in)，其中 N = P × H × W
            输出: (B, M, C_out)，其中 M = P' × H' × W', C_out = 2 × C_in

        Example:
            >>> downsample = DownSample3D(128, (13, 128, 256), (13, 64, 128))
            >>> x = torch.randn(4, 425984, 128)
            >>> out = downsample(x)
            >>> out.shape
            torch.Size([4, 106496, 256])

    """

    def __init__(self, in_dim, input_resolution, output_resolution):
        super().__init__()
        self.linear = nn.Linear(in_dim * 4, in_dim * 2, bias=False)
        self.norm = nn.LayerNorm(4 * in_dim)
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution

        in_pl, in_lat, in_lon = self.input_resolution
        out_pl, out_lat, out_lon = self.output_resolution

        h_pad = out_lat * 2 - in_lat
        w_pad = out_lon * 2 - in_lon

        pad_top = h_pad // 2
        pad_bottom = h_pad - pad_top

        pad_left = w_pad // 2
        pad_right = w_pad - pad_left

        pad_front = pad_back = 0

        self.pad = nn.ZeroPad3d(
            (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
        )

    def forward(self, x):
        B, N, C = x.shape
        in_pl, in_lat, in_lon = self.input_resolution
        out_pl, out_lat, out_lon = self.output_resolution
        x = x.reshape(B, in_pl, in_lat, in_lon, C)

        # Padding the input to facilitate downsampling
        x = self.pad(x.permute(0, -1, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        x = x.reshape(B, in_pl, out_lat, 2, out_lon, 2, C).permute(0, 1, 2, 4, 3, 5, 6)
        x = x.reshape(B, out_pl * out_lat * out_lon, 4 * C)

        x = self.norm(x)
        x = self.linear(x)
        return x


class DownSample2D(nn.Module):
    """
        通过空间合并和线性变换实现二维空间下采样。

        Args:
            in_dim (int): 输入通道数
            input_resolution (tuple[int, int]): 输入空间维度 (H, W)
            output_resolution (tuple[int, int]): 输出空间维度 (H', W')

        形状:
            输入: (B, N, C_in)，其中 N = H × W
            输出: (B, M, C_out)，其中 M = H' × W', C_out = 2 × C_in

        Example:
            >>> downsample = DownSample2D(64, (128, 256), (64, 128))
            >>> x = torch.randn(8, 32768, 64)
            >>> out = downsample(x)
            >>> out.shape
            torch.Size([8, 8192, 128])

    """

    def __init__(self, in_dim, input_resolution, output_resolution):
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
