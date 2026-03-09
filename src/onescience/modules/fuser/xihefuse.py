from collections.abc import Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .xihelocalsiefuser import XiheLocalSIEFuser
from .xiheglobalsiefuser import XiheGlobalSIEFuser


class XiheFuser(nn.Module):
    """
    Xihe 模型的海洋特定特征融合模块，包含局部和全局特征提取器。

    Args:
        dim (int): 输入通道数 C。
        input_resolution (tuple[int, int, int]): 输入空间分辨率 (Pl, Lat, Lon)。
        num_local (int): 局部 SIE (Local SIE) 模块的数量。
        num_heads_local (int): 局部 SIE 的注意力头数量，默认为 6。
        num_heads_global (int): 全局 SIE 的注意力头数量，默认为 12。
        window_size (tuple[int, int, int]): 局部 SIE 的 3D 窗口大小 (Wpl, Wlat, Wlon)，默认为 (1, 6, 12)。
        mlp_ratio (float): MLP 隐层扩展比例，默认为 4.0。
        qkv_bias (bool): 是否在 QKV 上添加可学习的偏置，默认为 True。
        drop_path (float): 随机深度（DropPath）比例，默认为 0.0。
        num_groups (int): 全局 SIE 的 group 数量 G，默认为 32。
        num_global (int): 全局 SIE (Global SIE) 模块的数量，默认为 1。
        depth_local (int): 每个局部 SIE 内 transformer block 的深度，默认为 2。
        norm_layer (nn.Module): 归一化层类型，默认为 nn.LayerNorm。

    形状:
        输入 obj: 包含以下属性的对象
            - x (torch.Tensor): 输入张量，形状为 (B, L, C)，其中 L = Pl × Lat × Lon
            - mask (torch.Tensor, optional): 海陆掩码，形状为 (B, L) 或可 reshape 为 (B, L)，1=有效（海洋），0=忽略（陆地）
        输出 x (torch.Tensor): 输出张量，形状与输入 x 相同，为 (B, L, C)

    Example:
        >>> fuser = XiheFuser(
        ...     dim=192,
        ...     input_resolution=(13, 128, 256),
        ...     num_local=2,
        ...     num_global=1,
        ...     num_heads_local=6,
        ...     num_heads_global=12,
        ...     window_size=(1, 6, 12)
        ... )
        >>> from types import SimpleNamespace
        >>> B, C, Pl, Lat, Lon = 2, 192, 13, 128, 256
        >>> L = Pl * Lat * Lon
        >>> x = torch.randn(B, L, C)
        >>> mask = torch.ones(B, L, dtype=torch.bool)  # 全海洋掩码
        >>> obj = SimpleNamespace(x=x, mask=mask)
        >>> out = fuser(obj)
        >>> out.shape
        torch.Size([2, 425984, 192])

    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_local,        #  Number of Local SIE 
        num_heads_local=6,
        num_heads_global=12,
        window_size=(1,6,12),
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        num_groups=32,
        num_global=1,       #  Number of Global SIE
        depth_local=2,      #  depth of transformer block
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim=dim
        self.num_local=num_local
        self.input_resolution=input_resolution


        # Local SIE modules
        self.local_sie_blocks = nn.ModuleList([
            XiheLocalSIEFuser(
                dim=dim,
                input_resolution=input_resolution,             
            )
            for _ in range(num_local)

            
        ])

        # Global SIE modules
        self.global_sie_blocks = nn.ModuleList([
            XiheGlobalSIEFuser(
                dim=dim,
            )
            for _ in range(num_global)
        ])

    def forward(self, obj):
        """
        x: (B, N, C)
        mask: (可选) ocean-land mask
        """
        x=obj.x
        mask=obj.mask
        
        # Local SIE(s)
        for local in self.local_sie_blocks:
            x = local(obj) if mask is None else local(obj)
            obj.x=x

        # Global SIE(s)
        for global_sie in self.global_sie_blocks:
            x = global_sie(obj) if mask is None else global_sie(obj)
            obj.x=x

        return x