from collections.abc import Sequence

from torch import nn

from onescience.modules.embedding.oneembedding import OneEmbedding
from onescience.modules.sample.onesample import OneSample
from onescience.modules.transformer.onetransformer import OneTransformer


class FengWuEncoder(nn.Module):
    """
        FengWu 模型的二维层次化编码器。

        该模块先用 `PanguEmbedding` 对二维气象场做 patch embedding，
        再在高分辨率 patch 网格上做若干层 `EarthTransformer2DBlock`，
        之后使用 `PanguDownSample` 下采样到中分辨率，并继续做中分辨率
        Transformer 编码。

        输出包括：

        - 中分辨率 token 序列
        - 高分辨率 skip 特征

        供 FengWu 主模型后续的多变量 3D fuser 和 decoder 使用。

        Args:
            input_resolution (tuple[int, int]):
                高分辨率 patch 网格尺寸 `(Height, Width)`。
            middle_resolution (tuple[int, int]):
                下采样后中分辨率 patch 网格尺寸 `(OutHeight, OutWidth)`。
            in_chans (int):
                输入气象变量通道数。
            img_size (tuple[int, int]):
                原始输入场尺寸 `(Height, Width)`。
            patch_size (tuple[int, int]):
                patch 切分尺寸 `(PatchHeight, PatchWidth)`。
            dim (int):
                高分辨率阶段的特征维度。
            depth (int):
                高分辨率 Transformer block 层数。
            depth_middle (int):
                中分辨率 Transformer block 层数。
            num_heads (int | tuple[int, int]):
                注意力头数配置；若为二元组，则顺序为
                `(HighResolutionHeads, MiddleResolutionHeads)`。
            window_size (int | tuple[int, int]):
                二维窗口大小。
            mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, norm_layer:
                标准 Transformer 配置项。

        形状:
            输入:
                `x` 形状为 `(Batch, in_chans, Height, Width)`
            输出:
                - `x` 形状为 `(Batch, middle_resolution[0] * middle_resolution[1], 2 * dim)`
                - `skip` 形状为 `(Batch, input_resolution[0], input_resolution[1], dim)`

        Example:
            >>> encoder = FengWuEncoder(
            ...     input_resolution=(181, 360),
            ...     middle_resolution=(91, 180),
            ...     in_chans=37,
            ...     img_size=(721, 1440),
            ...     patch_size=(4, 4),
            ...     dim=192,
            ... )
            >>> x = torch.randn(2, 37, 721, 1440)
            >>> out, skip = encoder(x)
            >>> out.shape
            torch.Size([2, 91 * 180, 192 * 2])
            >>> skip.shape
            torch.Size([2, 181, 360, 192])
    """

    def __init__(
        self,
        input_resolution=(181, 360),
        middle_resolution=(91, 180),
        in_chans=37,
        img_size=(721, 1440),
        patch_size=(4, 4),
        dim=192,
        depth=2,
        depth_middle=6,
        num_heads=(6, 12),
        window_size=(6, 12),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.depth_middle = depth_middle
        if isinstance(drop_path, Sequence):
            drop_path_middle = drop_path[depth:]
            drop_path = drop_path[:depth]
        else:
            drop_path_middle = drop_path
        if isinstance(num_heads, Sequence):
            num_heads_middle = num_heads[1]
            num_heads = num_heads[0]
        else:
            num_heads_middle = num_heads

        self.patchembed2d = OneEmbedding(
            style="PanguEmbedding",
            img_size=img_size,
            patch_size=patch_size,
            Variables=in_chans,
            embed_dim=dim,
        )
        self.blocks = nn.ModuleList(
            [
                OneTransformer(
                    style="EarthTransformer2DBlock",
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=(0, 0) if i % 2 == 0 else None,
                    drop_path=drop_path[i] if isinstance(drop_path, Sequence) else drop_path,

                )
                for i in range(depth)
            ]
        )

        self.downsample = OneSample(
            style="PanguDownSample",
            in_dim=dim,
            input_resolution=input_resolution,
            output_resolution=middle_resolution,
        )

        self.blocks_middle = nn.ModuleList(
            [
                OneTransformer(
                    style="EarthTransformer2DBlock",
                    dim=dim * 2,
                    input_resolution=middle_resolution,
                    num_heads=num_heads_middle,
                    window_size=window_size,
                    shift_size=(0, 0) if i % 2 == 0 else None,
                    drop_path=drop_path_middle[i] if isinstance(drop_path_middle, Sequence) else drop_path_middle,
                )
                for i in range(depth_middle)
            ]
        )

    def forward(self, x):
        x = self.patchembed2d(x)
        Batch, Channels, Height, Width = x.shape
        x = x.reshape(Batch, Channels, -1).transpose(1, 2)
        for blk in self.blocks:
            x = blk(x)
        skip = x.reshape(Batch, Height, Width, Channels)
        x = self.downsample(x)
        for blk in self.blocks_middle:
            x = blk(x)
        return x, skip
