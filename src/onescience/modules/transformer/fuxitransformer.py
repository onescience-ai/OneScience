import torch
from torch import nn
from timm.layers.helpers import to_2tuple
from timm.models.swin_transformer_v2 import SwinTransformerV2Stage

from onescience.modules.func_utils.fuxi_utils import get_pad2d
from onescience.modules.sample.onesample import OneSample


class FuxiTransformer(nn.Module):
    """
        Fuxi 模型的二维 trunk 模块。

        该模块采用：

        - `FuxiDownSample`
        - `SwinTransformerV2Stage`
        - `FuxiUpSample`

        构成 U 形主干结构。

        输入首先被下采样到较低分辨率，再在低分辨率网格上做多层
        `SwinTransformerV2Stage` 计算，最后与下采样输出做通道拼接并上采样
        回原分辨率。

        当分辨率与窗口大小不整除时，内部会先做 ZeroPad，再在 trunk 输出后做 crop。

        Args:
            embed_dim (int):
                输入与输出特征通道数。
            num_groups (int | tuple[int, int]):
                下采样与上采样模块中的 `GroupNorm` 分组数。
            input_resolution (tuple[int, int]):
                下采样后 trunk 网格尺寸 `(Height, Width)`。
            num_heads (int):
                Swin Transformer 的注意力头数。
            window_size (int | tuple[int, int]):
                局部窗口大小。
            depth (int):
                `SwinTransformerV2Stage` 的 block 层数。

        形状:
            输入:
                `x` 形状为 `(Batch, embed_dim, Height, Width)`
            输出:
                `x` 形状为 `(Batch, embed_dim, Height, Width)`

            补充说明：
            - 这里的输入 `Height` 与 `Width` 指的是 embedding 后、进入 trunk 前的二维特征图尺寸
            - `input_resolution` 则是下采样后送入 `SwinTransformerV2Stage` 的尺寸

        Examples:
            >>> Batch = 2
            >>> Height = 180
            >>> Width = 360
            >>> transformer = FuxiTransformer(
            ...     embed_dim=1536,
            ...     num_groups=32,
            ...     input_resolution=(90, 180),
            ...     num_heads=8,
            ...     window_size=7,
            ...     depth=48,
            ... )
            >>> x = torch.randn(Batch, 1536, Height, Width)
            >>> out = transformer(x)
            >>> out.shape
            torch.Size([2, 1536, 180, 360])
    """

    def __init__(
        self,
        embed_dim=1536,
        num_groups=32,
        input_resolution=(90, 180),
        num_heads=8,
        window_size=7,
        depth=48,
    ):
        super().__init__()

        num_groups = to_2tuple(num_groups)
        window_size = to_2tuple(window_size)
        Padding = get_pad2d(input_resolution, window_size)
        PaddingLeft, PaddingRight, PaddingTop, PaddingBottom = Padding
        self.padding = Padding
        self.pad = nn.ZeroPad2d(Padding)

        PaddedResolution = list(input_resolution)
        PaddedResolution[0] = PaddedResolution[0] + PaddingTop + PaddingBottom
        PaddedResolution[1] = PaddedResolution[1] + PaddingLeft + PaddingRight

        self.down = OneSample(
            style="FuxiDownSample",
            in_chans=embed_dim,
            out_chans=embed_dim,
            num_groups=num_groups[0],
        )
        self.layer = SwinTransformerV2Stage(
            embed_dim,
            embed_dim,
            PaddedResolution,
            depth,
            num_heads,
            window_size,
        )
        self.up = OneSample(
            style="FuxiUpSample",
            in_chans=embed_dim * 2,
            out_chans=embed_dim,
            num_groups=num_groups[0],
        )

    def forward(self, x):
        PaddingLeft, PaddingRight, PaddingTop, PaddingBottom = self.padding
        x = self.down(x)

        Shortcut = x

        x = self.pad(x)
        _, _, PaddedHeight, PaddedWidth = x.shape

        x = x.permute(0, 2, 3, 1)
        x = self.layer(x)
        x = x.permute(0, 3, 1, 2)

        x = x[
            :,
            :,
            PaddingTop : PaddedHeight - PaddingBottom,
            PaddingLeft : PaddedWidth - PaddingRight,
        ]

        x = torch.cat([Shortcut, x], dim=1)

        x = self.up(x)
        return x
