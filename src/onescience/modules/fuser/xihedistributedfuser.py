from collections.abc import Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .xihedistributedlocalsiefuser import XiheDistributedLocalSIEFuser
from .xihedistributedglobalsiefuser import XiheDistributedGlobalSIEFuser


class XiheDistributedFuser(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        num_local,
        num_heads_local=6,
        num_heads_global=12,
        window_size=(1, 6, 12),
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        num_groups=32,
        num_global=1,
        depth_local=2,
        norm_layer=nn.LayerNorm,
        config=None,
    ):
        super().__init__()
        self.dim = dim
        self.num_local = num_local
        self.input_resolution = input_resolution

        self.local_sie_blocks = nn.ModuleList([
            XiheDistributedLocalSIEFuser(
                dim=dim,
                input_resolution=input_resolution,
                depth=depth_local,
                num_heads=num_heads_local,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                config=config,
            )
            for _ in range(num_local)
        ])

        self.global_sie_blocks = nn.ModuleList([
            XiheDistributedGlobalSIEFuser(
                dim=dim,
                num_heads=num_heads_global,
                num_groups=num_groups,
                config=config,
            )
            for _ in range(num_global)
        ])

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

        for local in self.local_sie_blocks:
            x = local(obj) if mask is None else local(obj)
            obj["x"] = x

        for global_sie in self.global_sie_blocks:
            x = global_sie(obj) if mask is None else global_sie(obj)
            obj["x"] = x

        return x
