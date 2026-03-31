import torch
from torch import nn


class PanguUpSample(nn.Module):
    """
    Pangu-Weather 模型的统一上采样模块。

    该模块与 `PanguDownSample` 对称，支持二维和三维 token 网格输入：
    - 二维输入表示 surface 分支，形状为 `(B, H * W, in_dim)`；
    - 三维输入表示 upper-air 分支，形状为 `(B, Pl * H * W, in_dim)`。

    实现逻辑统一走三维分支：先将每个 token 在线性层中扩展到 `4 * out_dim`，
    再按 2x2 子像素方式恢复到更高分辨率的空间网格，最后通过中心裁剪对齐目标分辨率。

    Args:
        input_resolution (tuple[int, int] | tuple[int, int, int]):
            输入分辨率。
        output_resolution (tuple[int, int] | tuple[int, int, int]):
            输出分辨率。水平方向应满足 `out_h <= 2 * in_h` 且 `out_w <= 2 * in_w`。
            对三维输入，还要求 `out_pl <= in_pl`。
        in_dim (int): 输入 token 通道数。
        out_dim (int): 输出 token 通道数。

    Shape:
        - 输入: `(B, N, in_dim)`
        - 输出: `(B, M, out_dim)`
    """

    def __init__(self, input_resolution, output_resolution, in_dim=384, out_dim=192):
        super().__init__()

        if len(input_resolution) == 2:
            input_resolution = (1, *input_resolution)
        elif len(input_resolution) != 3:
            raise ValueError("input_resolution must have 2 or 3 dimensions")

        if len(output_resolution) == 2:
            output_resolution = (1, *output_resolution)
        elif len(output_resolution) != 3:
            raise ValueError("output_resolution must have 2 or 3 dimensions")

        self.input_resolution = input_resolution
        self.output_resolution = output_resolution
        self.in_dim = in_dim
        self.out_dim = out_dim

        in_pl, in_lat, in_lon = self.input_resolution
        out_pl, out_lat, out_lon = self.output_resolution

        if out_pl > in_pl:
            raise ValueError("output pressure levels must be less than or equal to input pressure levels")
        if out_lat > in_lat * 2 or out_lon > in_lon * 2:
            raise ValueError("output spatial resolution cannot exceed twice the input resolution")

        self.linear1 = nn.Linear(in_dim, out_dim * 4, bias=False)
        self.linear2 = nn.Linear(out_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, channels = x.shape
        in_pl, in_lat, in_lon = self.input_resolution
        out_pl, out_lat, out_lon = self.output_resolution

        expected_tokens = in_pl * in_lat * in_lon
        if num_tokens != expected_tokens:
            raise ValueError(f"Expected {expected_tokens} tokens, but received {num_tokens}")
        if channels != self.in_dim:
            raise ValueError(f"Expected input dim {self.in_dim}, but received {channels}")

        x = self.linear1(x)
        x = x.reshape(batch_size, in_pl, in_lat, in_lon, 2, 2, self.out_dim)
        x = x.permute(0, 1, 2, 4, 3, 5, 6)
        x = x.reshape(batch_size, in_pl, in_lat * 2, in_lon * 2, self.out_dim)

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
        x = x.reshape(batch_size, out_pl * out_lat * out_lon, self.out_dim)
        x = self.norm(x)
        x = self.linear2(x)
        return x
