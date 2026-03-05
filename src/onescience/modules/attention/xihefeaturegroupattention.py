import torch
import torch.nn as nn
from onescience.modules.func_utils import Mlp

#GLOBAL 1
class FeatureGroupingAttention(nn.Module):
    """
    全局 SIE 的第一步：特征分组模块，将局部特征聚合为少量的 group 表示。

    Args:
        dim (int): 输入通道数 C。
        num_groups (int): group vectors 数量 G，默认为 32。
        num_heads (int): 多头注意力的头数，默认为 12。
        qkv_bias (bool): 是否在 QKV 上添加可学习的偏置，默认为 True。
        attn_drop (float): 注意力权重的 dropout 比例，默认为 0.0。
        proj_drop (float): 输出投影的 dropout 比例，默认为 0.0。
        LN (nn.Module): 归一化层类型，默认为 nn.LayerNorm。
        drop_layer (nn.Module): dropout 层类型，默认为 nn.Dropout。

    形状:
        输入 obj: 包含以下属性的对象
            - x (torch.Tensor): 输入张量，形状为 (B, N, C)，其中 N = Pl × Lat × Lon
            - mask (torch.Tensor): 掩码张量，形状为 (B, N) 或 (B, 1, H, W) 或 (B, H, W)
        输出 G_prime (torch.Tensor): 聚合后的 group 特征，形状为 (B, G, C)

    Example:
        >>> grouping = FeatureGroupingAttention(
        ...     dim=192,
        ...     num_groups=32,
        ...     num_heads=12
        ... )
        >>> from types import SimpleNamespace
        >>> B, N, C = 2, 425984, 192  # N = 13*128*256
        >>> x = torch.randn(B, N, C)
        >>> mask = torch.ones(B, N, dtype=torch.bool)
        >>> obj = SimpleNamespace(x=x, mask=mask)
        >>> out = grouping(obj)
        >>> out.shape
        torch.Size([2, 32, 192])
    """

    def __init__(
        self,
        dim, 
        num_groups=32, 
        num_heads=12, 
        qkv_bias=True,
        attn_drop=0.0, 
        proj_drop=0.0,
        LN=nn.LayerNorm,
        drop_layer=nn.Dropout,
        ):
        super().__init__()
        self.dim = dim
        self.num_groups = num_groups  
        self.num_heads = num_heads  #自定义，分组多就可表示的更精细

        # 初始化 learnable group vectors (相当于 G_l)
        self.group_vectors = nn.Parameter(torch.randn(1, num_groups, dim))
        self.norm = LN(dim)
        # 多头注意力 (标准 vanilla Transformer Attention)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, bias=qkv_bias, batch_first=True
        )
        self.attn_drop = drop_layer(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = drop_layer(proj_drop)
    # def forward(self, x,mask_tokens=None):
    def forward(self,obj,mask=None):
        """
        x: (B, N, C)  -> 来自 Local SIE 的特征
        """
        x=obj.x
        mask_tokens=obj.mask
        
        B, N, C = x.shape
        x = self.norm(x)  # (B, N, C)
        
        #  expand group vectors (batch 内共享同一份 group 参数)
        G = self.group_vectors.expand(B, -1, -1)  # (B, G, C)
        # Multi-Head Cross-Attention
        if mask_tokens is None:
            return None
        if mask_tokens.dim() == 4:              # (B,1,H,W)
            mask_tokens = mask_tokens.squeeze(1)
        if mask_tokens.dim() == 3:              # (B,H,W)
            mask_tokens = mask_tokens.reshape(B, -1)
        assert mask_tokens.shape == (B, N)
        key_padding_mask = None if mask_tokens is None else (mask_tokens == 0)
        G_prime, _ = self.attn(query=G, key=x, value=x,key_padding_mask=key_padding_mask)
        #  输出更新后的 group vectors
        G_prime = self.proj_drop(self.proj(G_prime))  # (B, G, C)

        return G_prime