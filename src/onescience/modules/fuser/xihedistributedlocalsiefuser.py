from collections.abc import Sequence
import torch
import torch.nn as nn
from onescience.modules.func_utils import (
    Mlp, crop3d, get_pad3d, window_partition, window_reverse, DropPath,
)
from onescience.modules.attention.oneattention import OneAttention
from onescience.modules.transformer.onetransformer import OneTransformer


class XiheDistributedLocalSIEFuser(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        depth=2,
        num_heads=6,
        window_size=(1, 6, 12),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        config=None,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution

        self.blocks = nn.ModuleList(
            [
                OneTransformer(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    style="XiheDistributedLocalTransformer",
                    config=config,
                )
                for i in range(depth)
            ]
        )

    def forward(self, obj):
        if isinstance(obj, dict):
            x = obj["x"]
            mask = obj.get("mask")
            if mask is not None:
                mask = mask.clone().detach().float()
        else:
            x = obj.x
            mask = getattr(obj, 'mask', None)
            obj = {"x": x, "mask": mask}

        for blk in self.blocks:
            x = blk(x) if mask is None else blk(obj)
            obj["x"] = x
        return x
