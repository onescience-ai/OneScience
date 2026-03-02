from collections.abc import Sequence

import torch
from torch import nn

from onescience.modules.embedding.oneembedding import OneEmbedding
from onescience.modules.sample.onesample import OneSample
from onescience.modules.recovery.onerecovery import OneRecovery
from onescience.modules.transformer.onetransformer import OneTransformer

class FengWuDecoder(nn.Module):
    """A 2D Transformer Decoder Module for one stage

    Args:
        img_size (tuple[int]): image size(Lat, Lon).
        patch_size (tuple[int]): Patch token size of Patch Recovery.
        out_chans (int): number of output channels of Patch Recovery.
        dim (int): Number of input channels of transformer.
        output_resolution (tuple[int]): Input resolution for transformer after upsampling.
        middle_resolution (tuple[int]): Input resolution for transformer before upsampling.
        depth (int): Number of blocks for transformer after upsampling.
        depth_middle (int): Number of blocks for transformer before upsampling.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(
        self,
        input_resolution,
        output_resolution,
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
                    input_resolution=output_resolution,
                    num_heads=num_heads_middle,
                    window_size=window_size,
                    shift_size=(0, 0) if i % 2 == 0 else None,
                    drop_path=drop_path_middle[i] if isinstance(drop_path_middle, Sequence) else drop_path_middle,
                )
                for i in range(depth_middle)
            ]
        )

        self.downsample = OneSample(
                style="PanguDownSample2D",
                in_dim=dim,
                input_resolution=input_resolution,
                output_resolution=middle_resolution,
            )
        self.upsample = OneSample(
            style="PanguUpSample2D",
            in_dim=dim * 2,
            out_dim=dim,
            input_resolution=middle_resolution,
            output_resolution=output_resolution
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

        self.patchrecovery2d = OneRecovery(
            style="pangupatchrecovery2d",
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=2 * dim, 
            out_chans=out_chans
        )


    def forward(self, x, skip):
        B, Lat, Lon, C = skip.shape
        for blk in self.blocks_middle:
            x = blk(x)
        x = self.upsample(x)
        for blk in self.blocks:
            x = blk(x)
        output = torch.concat([x, skip.reshape(B, -1, C)], dim=-1)
        output = output.transpose(1, 2).reshape(B, -1, Lat, Lon)
        output = self.patchrecovery2d(output)
        return output