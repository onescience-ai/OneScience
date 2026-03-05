import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ==========================================
# 朴素线性注意力 (Simple / Vanilla)
# ==========================================

class Vanilla_Linear_Attention(nn.Module):
    """
    朴素线性注意力机制 (Simple Linear Attention)。

    该模块实现了一种计算复杂度为 O(N) 的高效注意力机制。
    通过利用矩阵乘法的结合律，即计算 Q(K^T V) 而非 (QK^T)V，它避免了显式计算 O(N^2) 的注意力矩阵。
    此版本是线性注意力的基础实现，特点是计算直接，通过除以序列长度或掩码有效长度进行归一化。
    它保留了特殊的 "先 Reshape 后 Linear" 的权重结构，以兼容特定的预训练权重。

    Args:
        dim (int): 输入特征维度。
        heads (int, optional): 注意力头数。默认值: 8。
        dim_head (int, optional): 每个注意力头的维度。默认值: 64。
        dropout (float, optional): Dropout 概率。默认值: 0.0。
        scale (float, optional): 自定义缩放因子。如果为 None，默认行为是除以序列长度 (1/N)。默认值: None。

    形状:
        输入 x: (B, N, C)，其中 C 必须等于 heads * dim_head。
        输入 mask (可选): (B, N) 或 (B, 1, 1, N) 的布尔掩码。True 表示有效 token，False 表示 padding。
        输出: (B, N, C)。

    Example:
        >>> attn = Vanilla_Linear_Attention(dim=128, heads=8, dim_head=16)
        >>> x = torch.randn(8, 100, 128)
        >>> # 创建掩码，假设后50个token是padding
        >>> mask = torch.ones(8, 100).bool()
        >>> mask[:, 50:] = False
        >>> out = attn(x, mask=mask)
        >>> out.shape
        torch.Size([8, 100, 128])
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., scale=None, **kwargs):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = scale
        self.dropout = nn.Dropout(dropout)
        
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        B, N, C = x.shape
        # [B, N, C] -> [B, H, N, D]
        x = x.reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous() 
        
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        if mask is not None:
            if mask.ndim == 2:
                mask = mask.view(B, 1, N, 1)
            k = k.masked_fill(~mask, 0.)
            v = v.masked_fill(~mask, 0.)
            denorm = mask.sum(dim=-2, keepdim=True).float()
        else:
            denorm = float(N)

        context = torch.matmul(k.transpose(-1, -2), v)
        
        if self.scale is not None:
            context = context * self.scale
        else:
            context = context / (denorm + 1e-8)

        context = self.dropout(context)
        res = torch.matmul(q, context) 
        res = rearrange(res, 'b h n d -> b n (h d)')
        return self.to_out(res)


# ==========================================
# 通用线性注意力 
# ==========================================

class LinearAttention(nn.Module):
    """
    通用线性注意力 (Generalized Linear Attention)。

    相比朴素版，它引入了更复杂的归一化机制（D^{-1}），通过特定的归一化项来处理 Query 和 Key 的加权和。
    它支持 'l1' (Softmax), 'l2', 'galerkin' 等多种归一化策略，在保持线性复杂度的同时提供了更好的数值稳定性和表达能力。
    此外，该模块支持交叉注意力（Cross Attention）和残差连接（GNOT 特有设计）。

    Args:
        dim (int): 输入特征维度。
        heads (int, optional): 注意力头数。默认值: 8。
        dim_head (int, optional): 每个注意力头的维度。默认值: 64。
        dropout (float, optional): Dropout 概率。默认值: 0.0。
        attn_type (str, optional): 归一化类型，支持 'l1', 'l2', 'galerkin'。默认值: 'l1'。

    形状:
        输入 x: (B, N, C)，Query 特征。
        输入 y (可选): (B, M, C)，Key/Value 特征。如果不提供，默认为自注意力 (y=x)。
        输入 mask (可选): (B, M) 或 (B, 1, M, 1) 的布尔掩码，用于屏蔽无效的 Key/Value。
        输出: (B, N, C)。

    Example:
        >>> l_attn = LinearAttention(dim=64, heads=8, dim_head=8, attn_type='l2')
        >>> x = torch.randn(4, 512, 64)
        >>> out = l_attn(x)
        >>> out.shape
        torch.Size([4, 512, 64])
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., attn_type='l1', **kwargs):
        super().__init__()
        self.n_head = heads
        self.dim_head = dim_head
        self.attn_type = attn_type
        
        self.key = nn.Linear(dim, dim)
        self.query = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, y=None, mask=None):
        """
        Args:
            x: Query [B, N, C]
            y: Key/Value [B, M, C] (Optional, default=x)
            mask: [B, M] (针对 Key/Value 的 mask)
        """
        y = x if y is None else y
        B, T1, C = x.size()
        _, T2, _ = y.size()
        
        # 投影与分头
        q = self.query(x).view(B, T1, self.n_head, self.dim_head).transpose(1, 2) # [B, H, T1, D]
        k = self.key(y).view(B, T2, self.n_head, self.dim_head).transpose(1, 2)   # [B, H, T2, D]
        v = self.value(y).view(B, T2, self.n_head, self.dim_head).transpose(1, 2) # [B, H, T2, D]
        if mask is not None:
            if mask.ndim == 2:
                mask = mask.view(B, 1, T2, 1) # [B, 1, M, 1]
            
            if self.attn_type in ['l1', 'galerkin']:
                k = k.masked_fill(~mask, -1e9) 
            elif self.attn_type == 'l2':
                k = k.masked_fill(~mask, 0.)
            
            v = v.masked_fill(~mask, 0.)

        # 归一化处理
        if self.attn_type == 'l1':
            q = q.softmax(dim=-1)
            k = k.softmax(dim=-1)
            
            if mask is not None:
                k = k.masked_fill(~mask, 0.)

            k_cumsum = k.sum(dim=-2, keepdim=True) # [B, H, 1, D]
            D_inv = 1. / (q * k_cumsum).sum(dim=-1, keepdim=True) 
            
        elif self.attn_type == "galerkin":
            q = q.softmax(dim=-1)
            k = k.softmax(dim=-1)
            if mask is not None:
                k = k.masked_fill(~mask, 0.)
                valid_len = mask.sum(dim=-2, keepdim=True)
                D_inv = 1. / (valid_len + 1e-8)
            else:
                D_inv = 1. / float(T2)
                
        elif self.attn_type == "l2":
            q = q / (q.norm(dim=-1, keepdim=True, p=1) + 1e-8)
            k = k / (k.norm(dim=-1, keepdim=True, p=1) + 1e-8)
            k_cumsum = k.sum(dim=-2, keepdim=True)
            D_inv = 1. / ((q * k_cumsum).abs().sum(dim=-1, keepdim=True) + 1e-8)
            
        else:
            raise NotImplementedError

        # 核心线性 Attention 计算: Q (K^T V)
        context = k.transpose(-2, -1) @ v  # [B, H, D, D]
        
        y = self.attn_drop((q @ context) * D_inv + q)

        y = rearrange(y, 'b h n d -> b n (h d)')
        y = self.proj(y)
        return y