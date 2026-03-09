import torch
from torch import nn

class PanguDownSample2D(nn.Module):
    """
        Pangu-Weather 风格的 2D 空间下采样模块。
        
        类似 Swin Transformer 的 Patch Merging 操作，将 2×2 相邻 token 在通道维度拼接
        （C → 4C），再经 LayerNorm 与线性投影还原通道数（4C → 2C），空间分辨率缩小为原来
        的 1/2。当输入分辨率不能被 2 整除时（如 lat=181），自动进行 ZeroPad 补齐。
        
        Args:
            input_resolution (tuple[int, int]): 输入特征图的空间分辨率 (lat, lon)。
            output_resolution (tuple[int, int]): 输出特征图的空间分辨率 (lat//2, lon//2)，
                应满足 output_resolution ≈ input_resolution / 2，模块内部会自动计算所需
                padding 量以对齐 output_resolution * 2 与 input_resolution 的差值。
            in_dim (int, optional): 输入 token 的通道数，输出通道数为 in_dim * 2，
                默认为 192。
        
        形状:
            - 输入 x: (B, lat * lon, C)，其中 C = in_dim
            - 输出:   (B, out_lat * out_lon, C * 2)
        
        Examples:
            >>> # 气象场分辨率 181×360 → 91×180 下采样
            >>> # h_pad = 91*2 - 181 = 1（底部补1行）
            >>> # w_pad = 180*2 - 360 = 0（无需补齐）
            >>> # 输入 token 数: 181 * 360 = 65160
            >>> # 输出 token 数:  91 * 180 = 16380
            >>> downsample = PanguDownSample2D(
            ...     input_resolution=(181, 360),
            ...     output_resolution=(91, 180),
            ...     in_dim=192,
            ... )
            >>> x = torch.randn(2, 65160, 192)  # (B, lat*lon, C)
            >>> out = downsample(x)
            >>> out.shape
            torch.Size([2, 16380, 384])
            
            >>> # 整除情况下无需 padding（如 128×256 → 64×128）
            >>> downsample2 = PanguDownSample2D(
            ...     input_resolution=(128, 256),
            ...     output_resolution=(64, 128),
            ...     in_dim=192,
            ... )
            >>> x2 = torch.randn(2, 32768, 192)  # (B, 128*256, C)
            >>> out2 = downsample2(x2)
            >>> out2.shape
            torch.Size([2, 8192, 384])
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
