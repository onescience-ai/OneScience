import torch
from torch import nn

from ..func_utils import (
    get_earth_position_index,
    trunc_normal_,
)


class EarthAttention3D(nn.Module):
    """
        三维 Earth Attention。

        该模块在三维窗口内执行多头自注意力，并叠加地球位置偏置。
        输入已经按窗口划分并整理为：

        - 经度方向窗口数量折叠进 batch 维
        - `(PressureLevels, Height)` 方向窗口数量合并为一维

        因此它处理的不是原始三维网格，而是窗口化后的 token 张量。

        Args:
            dim (int):
                输入与输出特征维度。
            input_resolution (tuple[int, int, int]):
                padding 后的三维网格尺寸 `(PressureLevels, Height, Width)`。
            window_size (tuple[int, int, int]):
                窗口大小 `(WindowPressureLevels, WindowHeight, WindowWidth)`。
            num_heads (int):
                多头注意力头数。
            qkv_bias, qk_scale, attn_drop, proj_drop:
                标准注意力配置项。

        形状:
            输入:
                `x` 形状为
                `(Batch * NumWidthWindows, NumPressureHeightWindows, WindowTokens, dim)`
            输出:
                `x` 形状为
                `(Batch * NumWidthWindows, NumPressureHeightWindows, WindowTokens, dim)`

        Example:
            >>> attn = EarthAttention3D(
            ...     dim=192,
            ...     input_resolution=(14, 128, 256),
            ...     window_size=(2, 8, 8),
            ...     num_heads=6,
            ... )
            >>> x = torch.randn(128, 112, 128, 192)
            >>> out = attn(x)
            >>> out.shape
            torch.Size([128, 112, 128, 192])
            >>> mask = torch.zeros(32, 112, 128, 128)
            >>> out = attn(x, mask=mask)
            >>> out.shape
            torch.Size([128, 112, 128, 192])
    """
    def __init__(
        self,
        dim,
        input_resolution,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wpl, Wlat, Wlon
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.num_pressure_height_windows = (input_resolution[0] // window_size[0]) * (
            input_resolution[1] // window_size[1]
        )

        self.earth_position_bias_table = nn.Parameter(
            torch.zeros(
                (window_size[0] ** 2)
                * (window_size[1] ** 2)
                * (window_size[2] * 2 - 1),
                self.num_pressure_height_windows,
                num_heads,
            )
        )

        earth_position_index = get_earth_position_index(window_size)
        self.register_buffer("earth_position_index", earth_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.earth_position_bias_table = trunc_normal_(
            self.earth_position_bias_table, std=0.02
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask=None):
        """
        Args:
            x:
                `(Batch * NumWidthWindows, NumPressureHeightWindows, WindowTokens, dim)`
            mask:
                `(NumWidthWindows, NumPressureHeightWindows, WindowTokens, WindowTokens)` 或 `None`
        """
        BatchTimesWidthWindows, NumPressureHeightWindows, WindowTokens, Channels = x.shape
        qkv = (
            self.qkv(x)
            .reshape(
                BatchTimesWidthWindows,
                NumPressureHeightWindows,
                WindowTokens,
                3,
                self.num_heads,
                Channels // self.num_heads,
            )
            .permute(3, 0, 4, 1, 2, 5)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        earth_position_bias = self.earth_position_bias_table[
            self.earth_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.num_pressure_height_windows,
            -1,
        )
        earth_position_bias = earth_position_bias.permute(3, 2, 0, 1).contiguous()
        attn = attn + earth_position_bias.unsqueeze(0)

        if mask is not None:
            NumWidthWindows = mask.shape[0]
            attn = attn.view(
                BatchTimesWidthWindows // NumWidthWindows,
                NumWidthWindows,
                self.num_heads,
                NumPressureHeightWindows,
                WindowTokens,
                WindowTokens,
            ) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(
                -1,
                self.num_heads,
                NumPressureHeightWindows,
                WindowTokens,
                WindowTokens,
            )
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 2, 3, 1, 4).reshape(
            BatchTimesWidthWindows,
            NumPressureHeightWindows,
            WindowTokens,
            Channels,
        )
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
