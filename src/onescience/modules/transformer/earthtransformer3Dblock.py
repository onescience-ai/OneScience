import torch
from torch import nn
from ..func_utils import DropPath, Mlp, get_pad3d, crop3d, window_partition, window_reverse, get_shift_window_mask

from ..attention.oneattention import OneAttention


class EarthTransformer3DBlock(nn.Module):
    """
        三维 Earth Transformer block。

        该模块是面向三维 patch 网格的局部窗口 Transformer block，
        内部结构为：

        - `LayerNorm`
        - 可选三维循环移位
        - `OneAttention(style="EarthAttention3D")`
        - 残差连接
        - `Mlp`
        - 残差连接

        窗口注意力在 `(PressureLevels, Height, Width)` 三维网格上执行，并带有
        地球位置偏置。当输入分辨率不能整除窗口大小时，模块会先 padding，再在输出端
        crop 回原尺寸。

        Args:
            dim (int):
                输入与输出 token 特征维度。
            input_resolution (tuple[int, int, int]):
                输入三维网格尺寸 `(PressureLevels, Height, Width)`。
            num_heads (int):
                多头注意力头数。
            window_size (tuple[int, int, int] | None):
                局部窗口大小 `(WindowPressureLevels, WindowHeight, WindowWidth)`。
            shift_size (tuple[int, int, int] | None):
                循环移位大小 `(ShiftPressureLevels, ShiftHeight, ShiftWidth)`。
                当为 `(0, 0, 0)` 时，不启用 shifted window。
            mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, act_layer, norm_layer:
                标准 Transformer 配置项。

        形状:
            输入:
                `x` 形状为 `(Batch, PressureLevels * Height * Width, dim)`
            输出:
                `x` 形状为 `(Batch, PressureLevels * Height * Width, dim)`

        Examples:
            >>> block_w = EarthTransformer3DBlock(
            ...     dim=192,
            ...     input_resolution=(13, 128, 256),
            ...     num_heads=6,
            ...     window_size=(2, 6, 12),
            ...     shift_size=(0, 0, 0),
            ... )
            >>> x = torch.randn(2, 13 * 128 * 256, 192)
            >>> out = block_w(x)
            >>> out.shape
            torch.Size([2, 425984, 192])

            >>> block_sw = EarthTransformer3DBlock(
            ...     dim=192,
            ...     input_resolution=(13, 128, 256),
            ...     num_heads=6,
            ...     window_size=(2, 6, 12),
            ...     shift_size=(1, 3, 6),
            ... )
            >>> out = block_sw(x)
            >>> out.shape
            torch.Size([2, 425984, 192])
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=None,
        shift_size=None,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        window_size = (2, 6, 12) if window_size is None else window_size
        shift_size = (1, 3, 6) if shift_size is None else shift_size
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        padding = get_pad3d(input_resolution, window_size)
        self.pad = nn.ZeroPad3d(padding)

        pad_resolution = list(input_resolution)
        pad_resolution[0] += padding[-1] + padding[-2]
        pad_resolution[1] += padding[2] + padding[3]
        pad_resolution[2] += padding[0] + padding[1]

        self.attn = OneAttention(
            style="EarthAttention3D",
            dim=dim,
            input_resolution=pad_resolution,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.use_roll = any(self.shift_size)

        if self.use_roll:
            attn_mask = get_shift_window_mask(pad_resolution, window_size, shift_size)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor):
        PressureLevels, Height, Width = self.input_resolution
        Batch, NumTokens, Channels = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(Batch, PressureLevels, Height, Width, Channels)

        x = self.pad(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        _, PaddedPressureLevels, PaddedHeight, PaddedWidth, _ = x.shape

        ShiftPressureLevels, ShiftHeight, ShiftWidth = self.shift_size
        if self.use_roll:
            shifted_x = torch.roll(
                x,
                shifts=(-ShiftPressureLevels, -ShiftHeight, -ShiftWidth),
                dims=(1, 2, 3),
            )
            x_windows = window_partition(shifted_x, self.window_size)
        else:
            shifted_x = x
            x_windows = window_partition(shifted_x, self.window_size)

        WindowPressureLevels, WindowHeight, WindowWidth = self.window_size
        x_windows = x_windows.view(
            x_windows.shape[0],
            x_windows.shape[1],
            WindowPressureLevels * WindowHeight * WindowWidth,
            Channels,
        )

        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = attn_windows.view(
            attn_windows.shape[0],
            attn_windows.shape[1],
            WindowPressureLevels,
            WindowHeight,
            WindowWidth,
            Channels,
        )

        if self.use_roll:
            shifted_x = window_reverse(
                attn_windows,
                self.window_size,
                Pl=PaddedPressureLevels,
                Lat=PaddedHeight,
                Lon=PaddedWidth,
            )
            x = torch.roll(
                shifted_x,
                shifts=(ShiftPressureLevels, ShiftHeight, ShiftWidth),
                dims=(1, 2, 3),
            )
        else:
            shifted_x = window_reverse(
                attn_windows,
                self.window_size,
                Pl=PaddedPressureLevels,
                Lat=PaddedHeight,
                Lon=PaddedWidth,
            )
            x = shifted_x

        x = crop3d(x.permute(0, 4, 1, 2, 3), self.input_resolution).permute(
            0, 2, 3, 4, 1
        )

        x = x.reshape(Batch, PressureLevels * Height * Width, Channels)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
