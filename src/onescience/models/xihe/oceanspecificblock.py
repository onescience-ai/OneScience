from collections.abc import Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from onescience.models.xihe.localsie import LocalSIE
from onescience.models.xihe.globalsie import GlobalSIE




class OceanSpecificBlock(nn.Module):
    """
    Ocean-Specific Block

    Args:
        dim (int): 输入特征通道数.
        input_resolution (tuple[int, int]): 输入特征分辨率(H, W).
        num_heads_local (int): Local SIE的注意力head数.
        num_heads_global (int): Global SIE的注意力head数.
        window_size (int): Local SIE 的窗口大小.
        mlp_ratio (float): MLP 隐层扩展比例.
        qkv_bias (bool): 是否使用 QKV bias.
        drop_path (float): DropPath 概率.
        num_groups (int): Global SIE 的 group 数量 G.
        num_local (int): Local SIE 块数.
        num_global (int): Global SIE 快数.
        depth_local (int): 每个 Local SIE 内 transformer block 深度.
        norm_layer (nn.Module): 归一化层类型（默认 LayerNorm).

    形状:
        输入:
            x: (B, N, C),token 序列(N = H x W)
            mask: (可选) (B, N) 或可 reshape 为 (B, N) 的海陆掩码(1=有效,0=忽略）
        输出:
            x: (B, N, C)，经过若干 Local SIE 与 Global SIE 后的 token 序列

    Returns:
        Tensor: 输出特征，形状为 (B, N, C)。
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
    



    