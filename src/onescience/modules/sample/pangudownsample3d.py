import torch
from torch import nn

class PanguDownSample3D(nn.Module):
    """
        Pangu-Weather 风格的 3D 大气变量下采样模块。
        
        PanguDownSample2D 的三维扩展版本，对 (气压层, 纬度, 经度) 三维特征图进行
        空间下采样。注意：仅对水平方向（纬度、经度）做 2x 下采样，气压层维度保持不变，
        因此输出的气压层数与输入相同（out_pl == in_pl）。
        
        Args:
            input_resolution (tuple[int, int, int]): 输入特征图的空间分辨率 (pl, lat, lon)。
            output_resolution (tuple[int, int, int]): 输出特征图的空间分辨率
                (pl, lat//2, lon//2)，气压层维度与输入保持一致，水平方向约为输入的 1/2，
                模块内部自动计算水平方向所需的 ZeroPad 量。
            in_dim (int, optional): 输入 token 的通道数，输出通道数为 in_dim * 2，
                默认为 192。
        
        形状:
            - 输入 x: (B, pl * lat * lon, C)，其中 C = in_dim
            - 输出:   (B, pl * out_lat * out_lon, C * 2)
        
        Examples:
            >>> # 典型 Pangu-Weather 大气变量配置
            >>> # 气压层保持不变: pl=8
            >>> # 水平分辨率 181×360 → 91×180
            >>> # h_pad = 91*2 - 181 = 1（底部补1行）
            >>> # w_pad = 180*2 - 360 = 0（无需补齐）
            >>> # 输入 token 数: 8 * 181 * 360 = 521280
            >>> # 输出 token 数: 8 *  91 * 180 = 131040
            >>> downsample = PanguDownSample3D(
            ...     input_resolution=(8, 181, 360),
            ...     output_resolution=(8, 91, 180),
            ...     in_dim=192,
            ... )
            >>> x = torch.randn(2, 521280, 192)  # (B, pl*lat*lon, C)
            >>> out = downsample(x)
            >>> out.shape
            torch.Size([2, 131040, 384])
            
            >>> # 整除情况下无需 padding（如 pl=13, 128×256 → 64×128）
            >>> downsample2 = PanguDownSample3D(
            ...     input_resolution=(13, 128, 256),
            ...     output_resolution=(13, 64, 128),
            ...     in_dim=192,
            ... )
            >>> x2 = torch.randn(2, 425984, 192)  # (B, 13*128*256, C)
            >>> out2 = downsample2(x2)
            >>> out2.shape
            torch.Size([2, 106496, 384])
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
