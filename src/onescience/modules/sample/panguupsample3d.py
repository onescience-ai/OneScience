import torch
from torch import nn

class PanguUpSample3D(nn.Module):
    """
        Pangu-Weather 风格的 3D 大气变量上采样模块。

        PanguDownSample3D 的逆操作，同时也是 PanguUpSample2D 的三维扩展版本。
        仅对水平方向（纬度、经度）做 2x 上采样，气压层维度通过切片直接对齐
        output_resolution 的 out_pl，不参与上采样计算。

        Args:
            in_dim (int, optional): 输入 token 的通道数，默认为 192 * 2 = 384。
            out_dim (int, optional): 输出 token 的通道数，线性层先扩展至 out_dim * 4，
                拆分后每个子像素通道数恢复为 out_dim，默认为 192。
            input_resolution (tuple[int, int, int]): 输入特征图的空间分辨率 (pl, lat, lon)。
            output_resolution (tuple[int, int, int]): 目标输出分辨率 (out_pl, out_lat, out_lon)，
                水平方向应满足 out_lat ≤ in_lat * 2 且 out_lon ≤ in_lon * 2，
                超出部分通过中心裁剪去除；气压层直接取前 out_pl 层。

        形状:
            - 输入 x: (B, pl * lat * lon, C)，其中 C = in_dim
            - 输出:   (B, out_pl * out_lat * out_lon, out_dim)

        Examples:
            >>> # 典型 Pangu-Weather 大气变量配置
            >>> # 气压层保持不变: pl=8
            >>> # 水平分辨率 91×180 → 181×360（对应 PanguDownSample3D 的逆操作）
            >>> # in_lat * 2 = 91 * 2 = 182，裁剪掉多余的1行: pad_h = 182 - 181 = 1
            >>> # in_lon * 2 = 180 * 2 = 360，无需裁剪: pad_w = 0
            >>> # 输入 token 数: 8 *  91 * 180 = 131040
            >>> # 输出 token 数: 8 * 181 * 360 = 521280
            >>> upsample = PanguUpSample3D(
            ...     in_dim=384,
            ...     out_dim=192,
            ...     input_resolution=(8, 91, 180),
            ...     output_resolution=(8, 181, 360),
            ... )
            >>> x = torch.randn(2, 131040, 384)  # (B, pl*lat*lon, C)
            >>> out = upsample(x)
            >>> out.shape
            torch.Size([2, 521280, 192])

            >>> # 整除情况下无需裁剪（如 pl=13, 64×128 → 128×256）
            >>> upsample2 = PanguUpSample3D(
            ...     in_dim=384,
            ...     out_dim=192,
            ...     input_resolution=(13, 64, 128),
            ...     output_resolution=(13, 128, 256),
            ... )
            >>> x2 = torch.randn(2, 106496, 384)  # (B, 13*64*128, C)
            >>> out2 = upsample2(x2)
            >>> out2.shape
            torch.Size([2, 425984, 192])
    """
    def __init__(self, 
                 in_dim=192*2, 
                 out_dim=192,
                 input_resolution=None, 
                 output_resolution=None,
                 ):
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