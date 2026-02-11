import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.layers import trunc_normal_
from einops import rearrange, repeat
from torch import einsum
from functools import partial, reduce

ACTIVATION = {
    'gelu': nn.GELU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU(0.1),
    'softplus': nn.Softplus,
    'ELU': nn.ELU,
    'silu': nn.SiLU
}


class MLP(nn.Module):
    """
    多层感知机（Multi-Layer Perceptron）。

    这是一个标准的深度前馈神经网络模块。它由一个预处理线性层、若干个中间层（支持残差连接）和一个后处理线性层组成。
    该模块设计灵活，支持多种激活函数和可配置的层数，适用于特征变换、投影等多种任务。

    Args:
        n_input (int): 输入特征的维度。
        n_hidden (int): 隐藏层的特征维度。
        n_output (int): 输出特征的维度。
        n_layers (int, optional): 中间隐藏层的层数。默认值: 1。
        act (str, optional): 激活函数类型，支持 'gelu', 'tanh', 'sigmoid', 'relu', 'leaky_relu', 'softplus', 'ELU', 'silu'。默认值: 'gelu'。
        res (bool, optional): 是否在中间层使用残差连接（Residual Connection）。默认值: True。

    形状:
        输入 x: (B, ..., N_in)，任意维度的张量，最后一维为输入特征维度。
        输出: (B, ..., N_out)，形状与输入相同，仅最后一维变为输出特征维度。

    Example:
        >>> mlp = MLP(n_input=64, n_hidden=128, n_output=64, n_layers=2)
        >>> x = torch.randn(8, 10, 64)
        >>> out = mlp(x)
        >>> out.shape
        torch.Size([8, 10, 64])
    """
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class PreNorm(nn.Module):
    """
        预归一化模块（Pre-Normalization）。

        

        在应用核心函数（如 Attention 或 MLP）之前，先对输入应用 LayerNorm。
        这种结构通常比 Post-Norm 更利于深层网络的训练稳定性，是现代 Transformer 架构（如 GPT-3, ViT）中的标准做法。

        Args:
            dim (int): 归一化层的维度。
            fn (nn.Module): 需要在归一化后执行的神经网络模块。

        形状:
            输入: 与 fn 的输入形状一致。
            输出: 与 fn 的输出形状一致。

        Example:
            >>> block = PreNorm(64, nn.Linear(64, 64))
            >>> x = torch.randn(10, 64)
            >>> out = block(x)
    """
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Attention(nn.Module):
    """
        标准的多头自注意力机制（Multi-Head Self-Attention）。

        

        实现了 O(N^2) 复杂度的点积注意力。
        计算公式为 Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V。
        该模块包含线性投影（Q, K, V）和输出投影。

        Args:
            dim (int): 输入特征维度。
            heads (int, optional): 注意力头数。默认值: 8。
            dim_head (int, optional): 每个注意力头的维度。默认值: 64。
            dropout (float, optional): Dropout 概率。默认值: 0.0。

        形状:
            输入 x: (B, N, C)，其中 B 是批次大小，N 是序列长度，C 是特征维度。
            输出: (B, N, C)。

        Example:
            >>> attn = Attention(dim=64, heads=4, dim_head=16)
            >>> x = torch.randn(8, 100, 64)
            >>> out = attn(x)
            >>> out.shape
            torch.Size([8, 100, 64])
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., **kwargs):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # B N C
        B, N, C = x.shape
        x = x.reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()  # B H N C
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        res = torch.matmul(attn, v)  # B H G D
        res = rearrange(res, 'b h n d -> b n (h d)')
        return self.to_out(res)

class FlashAttention(nn.Module):
    """
        FlashAttention 模块。

        利用 PyTorch 的 F.scaled_dot_product_attention 实现的高效注意力机制。
        它通常利用底层的显存优化（如 Tiling）来减少 HBM 读写次数，从而加速计算并减少显存占用。
        在现代 GPU 上通常比标准 Attention 更快。

        Args:
            dim (int): 输入特征维度。
            heads (int, optional): 注意力头数。默认值: 8。
            dim_head (int, optional): 每个注意力头的维度。默认值: 64。
            dropout (float, optional): Dropout 概率。默认值: 0.0。

        形状:
            输入 x: (B, N, C)，其中 B 是批次大小，N 是序列长度，C 是特征维度。
            输出: (B, N, C)。

        Example:
            >>> attn = FlashAttention(dim=64, heads=8)
            >>> x = torch.randn(8, 128, 64)
            >>> out = attn(x)
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., **kwargs):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dropout = nn.Dropout(dropout)
        
        # Separate projection layers for query, key, and value
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len, dim]
        batch_size, seq_len, _ = x.shape
        
        # Get query, key, value projections for all heads
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        
        # Reshape for multi-head attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)
        
        # Flash attention implementation
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        out = rearrange(attn_output, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Vanilla_Linear_Attention(nn.Module):
    """
        朴素线性注意力机制。

        通过改变计算顺序（先计算 K^T V）来实现 O(N) 的线性复杂度。
        该实现不对 Q 和 K 进行 Softmax 归一化，而是直接计算点积并除以序列长度 N 进行平均。
        这种方法在长序列任务中可以显著降低计算成本。

        Args:
            dim (int): 输入特征维度。
            heads (int, optional): 注意力头数。默认值: 8。
            dim_head (int, optional): 每个注意力头的维度。默认值: 64。
            dropout (float, optional): Dropout 概率。默认值: 0.0。

        形状:
            输入 x: (B, N, C)，其中 B 是批次大小，N 是序列长度，C 是特征维度。
            输出: (B, N, C)。

        Example:
            >>> lin_attn = Vanilla_Linear_Attention(dim=64)
            >>> x = torch.randn(8, 1024, 64)
            >>> out = lin_attn(x)
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., **kwargs):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # B N C
        B, N, C = x.shape
        x = x.reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()  # B H N C
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        dots = torch.matmul(k.transpose(-1, -2), v) / float(N)
        dots = self.dropout(dots)
        res = torch.matmul(q, dots)  # B H G D
        res = rearrange(res, 'b h n d -> b n (h d)')
        return self.to_out(res)


class LinearAttention(nn.Module):
    """
        一种更通用的线性注意力实现（源自 GNOT/MmGPT）。

        

        支持多种注意力归一化方式（如 l1, l2, galerkin）。
        它通过特定的归一化项 D^{-1} 来处理 Query 和 Key 的加权和，在保持线性复杂度的同时提供数值稳定性。
        该模块还支持交叉注意力（Cross Attention）。

        Args:
            dim (int): 输入特征维度。
            heads (int, optional): 注意力头数。默认值: 8。
            dim_head (int, optional): 每个注意力头的维度。默认值: 64。
            dropout (float, optional): Dropout 概率。默认值: 0.0。
            attn_type (str, optional): 注意力归一化类型，支持 'l1', 'l2', 'galerkin'。默认值: 'l1'。

        形状:
            输入 x: (B, N, C)。
            输入 y (可选): (B, M, C) 用于交叉注意力。如果不提供，则默认为自注意力 (y=x)。
            输出: (B, N, C)。

        Example:
            >>> l_attn = LinearAttention(dim=64, attn_type='l2')
            >>> x = torch.randn(4, 512, 64)
            >>> out = l_attn(x)
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0., attn_type='l1', **kwargs):
        super(LinearAttention, self).__init__()
        self.key = nn.Linear(dim, dim)
        self.query = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        # regularization
        self.attn_drop = nn.Dropout(dropout)
        # output projection
        self.proj = nn.Linear(dim, dim)
        self.n_head = heads
        self.dim_head = dim_head
        self.attn_type = attn_type

    def forward(self, x, y=None):
        y = x if y is None else y
        B, T1, C = x.size()
        _, T2, _ = y.size()
        q = self.query(x).view(B, T1, self.n_head, self.dim_head).transpose(1, 2)  # (B, nh, T, hs)
        k = self.key(y).view(B, T2, self.n_head, self.dim_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(y).view(B, T2, self.n_head, self.dim_head).transpose(1, 2)  # (B, nh, T, hs)

        if self.attn_type == 'l1':
            q = q.softmax(dim=-1)
            k = k.softmax(dim=-1)
            k_cumsum = k.sum(dim=-2, keepdim=True)
            D_inv = 1. / (q * k_cumsum).sum(dim=-1, keepdim=True)  # normalized
        elif self.attn_type == "galerkin":
            q = q.softmax(dim=-1)
            k = k.softmax(dim=-1)
            D_inv = 1. / T2
        elif self.attn_type == "l2":  # still use l1 normalization
            q = q / q.norm(dim=-1, keepdim=True, p=1)
            k = k / k.norm(dim=-1, keepdim=True, p=1)
            k_cumsum = k.sum(dim=-2, keepdim=True)
            D_inv = 1. / (q * k_cumsum).abs().sum(dim=-1, keepdim=True)  # normalized
        else:
            raise NotImplementedError

        context = k.transpose(-2, -1) @ v
        y = self.attn_drop((q @ context) * D_inv + q)

        # output projection
        y = rearrange(y, 'b h n d -> b n (h d)')
        y = self.proj(y)
        return y

def exists(val):
    return val is not None

def default(value, d):
    return d if not exists(value) else value

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

def linear_attn(q, k, v, kv_mask = None):
    dim = q.shape[-1]

    if exists(kv_mask):
        mask_value = max_neg_value(q)
        mask = kv_mask[:, None, :, None]
        k = k.masked_fill_(~mask, mask_value)
        v = v.masked_fill_(~mask, 0.)
        del mask

    q = q.softmax(dim=-1)
    k = k.softmax(dim=-2)

    q = q * dim ** -0.5

    context = einsum('bhnd,bhne->bhde', k, v)
    attn = einsum('bhnd,bhde->bhne', q, context)
    return attn.reshape(*q.shape)

def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(None, index))
    r = (*pre_slices, slice(index, None))
    return t[l], t[r]

class SelfAttention(nn.Module):
    """
        基于高效计算的自注意力封装。

        该模块内部使用了 linear_attn 函数（对 Q 和 K 进行 Softmax 归一化后计算 QK^TV）。
        它设计了 split 机制，虽然在默认配置下主要执行全局线性注意力，但其架构允许扩展为局部/全局混合注意力。

        Args:
            dim (int): 输入特征维度。
            heads (int): 注意力头数。
            dim_head (int, optional): 每个注意力头的维度。如果为 None，则默认为 dim // heads。
            dropout (float, optional): Dropout 概率。默认值: 0.0。

        形状:
            输入 x: (B, N, C)。
            输出: (B, N, C)。

        Example:
            >>> sa = SelfAttention(dim=64, heads=8)
            >>> x = torch.randn(8, 256, 64)
            >>> out = sa(x)
    """
    def __init__(self, dim, heads, dim_head = None,dropout = 0.):
        super().__init__()
        assert dim_head or (dim % heads) == 0, 'embedding dimension must be divisible by number of heads'
        d_heads = default(dim_head, dim // heads)

        self.heads = heads
        self.d_heads = d_heads

        self.global_attn_heads = heads
        self.global_attn_fn = linear_attn
        

        self.to_q = nn.Linear(dim, d_heads * heads, bias = False)

        kv_heads = heads

        self.kv_heads = kv_heads
        self.to_k = nn.Linear(dim, d_heads * kv_heads, bias = False)
        self.to_v = nn.Linear(dim, d_heads * kv_heads, bias = False)

        self.to_out = nn.Linear(d_heads * heads, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        q, k, v = (self.to_q(x), self.to_k(x), self.to_v(x))

        b, t, e, h, dh = *q.shape, self.heads, self.d_heads

        merge_heads = lambda x: x.reshape(*x.shape[:2], -1, dh).transpose(1, 2)

        q, k, v = map(merge_heads, (q, k, v))

        out = []

        split_index_fn = partial(split_at_index, 1, 0)

        (lq, q), (lk, k), (lv, v) = map(split_index_fn, (q, k, v))

        _, has_global = map(lambda x: x.shape[1] > 0, (lq, q))

        if has_global:
            global_out = self.global_attn_fn(q, k, v)
            out.append(global_out)

        attn = torch.cat(out, dim=1)
        attn = attn.transpose(1, 2).reshape(b, t, -1)
        return self.dropout(self.to_out(attn))
