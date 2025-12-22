from collections.abc import Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from ..layers.transformer_layers import Transformer3DBlock
from ..layers.mlp_layers import Mlp
from collections.abc import Sequence
from timm.layers import to_2tuple
from timm.models.swin_transformer import SwinTransformerStage
from onescience.models.xihe.localsie import LocalSIE
from onescience.models.xihe.globalsie import GlobalSIE

from ..utils import (
    PatchEmbed2D,
    PatchRecovery2D,
    crop2d,
    crop3d,
    get_pad2d,
    get_pad3d,
    get_shift_window_mask,
    window_partition,
    window_reverse,
)


class OceanSpecificBlock(nn.Module):
    """
    Ocean-Specific Block
    ---------------------
    Block1 & Block5: 1 Local + 1 Global
    Block2-Block4 : 2 Local + 1 Global
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads_local,
        num_heads_global,
        window_size,
        mlp_ratio,
        qkv_bias=True,
        drop_path=0.0,
        num_groups=32,
        num_local=1,        #  Number of Local SIE 
        num_global=1,       #  Number of Global SIE
        depth_local=2,      #  depth of transformer block
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim=dim
        self.num_groups=num_groups
        self.num_local=num_local
        self.num_global=num_global
        self.num_heads_local=num_heads_local    
        self.num_heads_global=num_heads_global
        self.window_size=window_size
        self.drop_path=drop_path

        # Local SIE modules
        self.local_sie_blocks = nn.ModuleList([
            LocalSIE(
                dim=dim,
                input_resolution=input_resolution,
                depth=depth_local,
                num_heads=num_heads_local,
                window_size=window_size,
                mlp_ratio=4.0,
                qkv_bias=True,
                drop_path=drop_path,
                norm_layer=norm_layer,                
            )
            for _ in range(num_local)
        ])

        # Global SIE modules
        self.global_sie_blocks = nn.ModuleList([
            GlobalSIE(
                dim=dim,
                num_heads=num_heads_global,
                num_groups=num_groups,
                norm_layer=norm_layer,
            )
            for _ in range(num_global)
        ])

    def forward(self, x, mask=None):
        """
        x: (B, N, C)
        mask: (可选) ocean-land mask
        """
        # Local SIE(s)
        for local in self.local_sie_blocks:
            x = local(x) if mask is None else local(x, mask=mask)

        # Global SIE(s)
        for global_sie in self.global_sie_blocks:
            x = global_sie(x) if mask is None else global_sie(x, mask=mask)

        return x
    



    