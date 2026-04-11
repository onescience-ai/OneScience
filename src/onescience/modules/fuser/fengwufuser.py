from collections.abc import Sequence

from torch import nn

from onescience.modules.transformer.onetransformer import OneTransformer


class FengWuFuser(nn.Module):
    """
        FengWu 模型的三维特征融合模块。

        该模块在三维网格 `(Variables, Height, Width)` 上堆叠多层
        `EarthTransformer3DBlock`，用于融合不同变量分支在中分辨率上的特征。

        输入通常来自多个 `FengWuEncoder` 的输出，在 FengWu 主模型中会先按变量轴
        拼接成统一三维网格，再展平成 token 序列送入该模块。

        Args:
            input_resolution (tuple[int, int, int]):
                三维输入网格尺寸 `(Variables, Height, Width)`。
            dim (int):
                输入与输出 token 特征维度。
            depth (int):
                3D Transformer block 层数。
            num_heads (int):
                多头自注意力头数。
            window_size (tuple[int, int, int]):
                三维窗口大小 `(VariablesWindow, HeightWindow, WidthWindow)`。
            mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, norm_layer:
                标准 Transformer 配置项。

        形状:
            输入:
                `x` 形状为 `(Batch, Variables * Height * Width, dim)`
            输出:
                `x` 形状为 `(Batch, Variables * Height * Width, dim)`

        Example:
            >>> fuser = FengWuFuser(
            ...     input_resolution=(6, 91, 180),
            ...     dim=192 * 2,
            ...     depth=6,
            ...     num_heads=12,
            ...     window_size=(2, 6, 12),
            ... )
            >>> Batch = 2
            >>> Variables, Height, Width, Channels = 6, 91, 180, 192 * 2
            >>> x = torch.randn(Batch, Variables * Height * Width, Channels)
            >>> out = fuser(x)
            >>> out.shape
            torch.Size([2, Variables * Height * Width, Channels])
    """

    def __init__(
        self,
        input_resolution=(6, 91, 180),
        dim=192 * 2,
        depth=6,
        num_heads=12,
        window_size=(2, 6, 12),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=[0.2] * 6,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        self.blocks = nn.ModuleList(
            [
                OneTransformer(
                    style="EarthTransformer3DBlock",
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=(0, 0, 0) if i % 2 == 0 else None,
                    drop_path=drop_path[i] if isinstance(drop_path, Sequence) else drop_path,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x
