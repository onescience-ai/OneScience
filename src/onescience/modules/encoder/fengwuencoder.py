from collections.abc import Sequence

import torch
from torch import nn

from onescience.modules.embedding.oneembedding import OneEmbedding
from onescience.modules.sampler.onesample import OneSample
from onescience.modules.transformer.onetransformer import OneTransformer

class FengWuEncoder(nn.Module):
    """A 2D Transformer Encoder Module for one stage

    Args:
        img_size (tuple[int]): image size(Lat, Lon).
        patch_size (tuple[int]): Patch token size of Patch Embedding.
        in_chans (int): number of input channels of Patch Embedding.
        dim (int): Number of input channels of transformer.
        input_resolution (tuple[int]): Input resolution for transformer before downsampling.
        middle_resolution (tuple[int]): Input resolution for transformer after downsampling.
        depth (int): Number of blocks for transformer before downsampling.
        depth_middle (int): Number of blocks for transformer after downsampling.
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
        middle_resolution,
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
            style="PanguEmbedding2D",
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
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
            style="PanguDownSample2D",
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
        B, C, Lat, Lon = x.shape
        x = x.reshape(B, C, -1).transpose(1, 2)
        for blk in self.blocks:
            x = blk(x)
        skip = x.reshape(B, Lat, Lon, C)
        x = self.downsample(x)
        for blk in self.blocks_middle:
            x = blk(x)
        return x, skip