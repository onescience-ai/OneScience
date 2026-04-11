from collections.abc import Sequence

import torch
from torch import nn

from onescience.modules.recovery.onerecovery import OneRecovery
from onescience.modules.sample.onesample import OneSample
from onescience.modules.transformer.onetransformer import OneTransformer


class FengWuDecoder(nn.Module):
    """
        FengWu 模型的解码器模块。

        该模块接收：

        - 中分辨率 token 序列
        - 高分辨率 skip 特征

        并按以下顺序完成解码：

        - 中分辨率 `EarthTransformer2DBlock`
        - `PanguUpSample`
        - 高分辨率 `EarthTransformer2DBlock`
        - 与 skip 特征沿通道维拼接
        - `PanguPatchRecovery`

        Args:
            output_resolution (tuple[int, int]):
                高分辨率 patch 网格尺寸 `(Height, Width)`。
            middle_resolution (tuple[int, int]):
                中分辨率 patch 网格尺寸 `(Height, Width)`。
            out_chans (int):
                最终输出变量通道数。
            img_size (tuple[int, int]):
                原始输出场尺寸 `(Height, Width)`。
            patch_size (tuple[int, int]):
                patch 恢复尺寸 `(PatchHeight, PatchWidth)`。
            dim (int):
                高分辨率特征维度；中分辨率阶段使用 `2 * dim`。
            depth (int):
                高分辨率 Transformer block 层数。
            depth_middle (int):
                中分辨率 Transformer block 层数。
            num_heads (tuple[int, int] | int):
                注意力头数配置；若为二元组，则顺序为
                `(HighResolutionHeads, MiddleResolutionHeads)`。
            window_size (tuple[int, int]):
                二维窗口大小。
            mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, norm_layer:
                标准 Transformer 配置项。

        形状:
            输入:
                - `inp[0]`: `(Batch, middle_resolution[0] * middle_resolution[1], 2 * dim)`
                - `inp[1]`: `(Batch, output_resolution[0], output_resolution[1], dim)`
            输出:
                `(Batch, out_chans, img_size[0], img_size[1])`

        Examples:
            >>> decoder = FengWuDecoder(
            ...     output_resolution=(181, 360),
            ...     middle_resolution=(91, 180),
            ...     out_chans=37,
            ...     img_size=(721, 1440),
            ...     patch_size=(4, 4),
            ...     dim=192,
            ...     depth=2,
            ...     depth_middle=6,
            ...     num_heads=(6, 12),
            ...     window_size=(6, 12),
            ... )
            >>> Batch = 2
            >>> NumTokens = 91 * 180
            >>> x = torch.randn(Batch, NumTokens, 384)
            >>> skip = torch.randn(Batch, 181, 360, 192)
            >>> out = decoder([x, skip])
            >>> out.shape
            torch.Size([2, 37, 721, 1440])
    """

    def __init__(
        self,
        output_resolution=(181, 360),
        middle_resolution=(91, 180),
        out_chans=37,
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
        self.out_chans = out_chans
        self.dim = dim
        self.output_resolution = output_resolution
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

        self.upsample = OneSample(
            style="PanguUpSample",
            in_dim=dim * 2,
            out_dim=dim,
            input_resolution=middle_resolution,
            output_resolution=output_resolution,
        )

        self.blocks = nn.ModuleList(
            [
                OneTransformer(
                    style="EarthTransformer2DBlock",
                    dim=dim,
                    input_resolution=output_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=(0, 0) if i % 2 == 0 else None,
                    drop_path=drop_path[i] if isinstance(drop_path, Sequence) else drop_path,

                )
                for i in range(depth)
            ]
        )

        self.patchrecovery2d = OneRecovery(
            style="PanguPatchRecovery",
            img_size=img_size,
            patch_size=patch_size,
            in_chans=2 * dim,
            out_chans=out_chans,
        )

    def forward(self, inp):
        x, skip = inp[0], inp[1]
        Batch, Height, Width, Channels = skip.shape
        for blk in self.blocks_middle:
            x = blk(x)
        x = self.upsample(x)
        for blk in self.blocks:
            x = blk(x)
        output = torch.concat([x, skip.reshape(Batch, -1, Channels)], dim=-1)
        output = output.transpose(1, 2).reshape(Batch, -1, Height, Width)
        output = self.patchrecovery2d(output)
        return output
