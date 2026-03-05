import torch
import torch.nn as nn
from functools import partial
from torch import einsum

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
    基于高效计算的自注意力封装 (增强版)。

    该模块是对 linear_attn 的高层封装。
    它保留了原始代码独特的 Split 机制（将头分为局部/全局两部分，尽管默认配置下全部为全局），
    并修复了原版无法传递 Mask 的问题。

    增强特性：
    1. Mask 支持：允许传入 (B, N) 的掩码，防止 Padding 干扰全局上下文聚合。
    2. 自定义 Scale：允许覆盖默认的 1/sqrt(dim) 缩放因子。

    Args:
        dim (int): 输入特征维度。
        heads (int): 注意力头数。
        dim_head (int, optional): 每个注意力头的维度。如果为 None，则默认为 dim // heads。
        dropout (float, optional): Dropout 概率。默认值: 0.0。
        scale (float, optional): 自定义缩放因子。如果为 None，默认使用 dim_head ** -0.5。

    形状:
        输入 x: (B, N, C)。
        输入 mask (可选): (B, N) 的布尔掩码。
        输出: (B, N, C)。

    Example:
        >>> sa = SelfAttention(dim=64, heads=8)
        >>> x = torch.randn(8, 256, 64)
        >>> mask = torch.ones(8, 256).bool() # 假设全有效
        >>> out = sa(x, mask=mask)
    """
    def __init__(self, dim, heads, dim_head=None, dropout=0., scale=None):
        super().__init__()
        assert dim_head or (dim % heads) == 0, 'embedding dimension must be divisible by number of heads'
        
        d_heads = default(dim_head, dim // heads)

        self.heads = heads
        self.d_heads = d_heads
        
        self.default_scale = d_heads ** -0.5
        self.custom_scale = scale
        
        self.global_attn_heads = heads
        self.global_attn_fn = linear_attn 

        self.to_q = nn.Linear(dim, d_heads * heads, bias=False)
        self.kv_heads = heads
        self.to_k = nn.Linear(dim, d_heads * heads, bias=False)
        self.to_v = nn.Linear(dim, d_heads * heads, bias=False)

        self.to_out = nn.Linear(d_heads * heads, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x (Tensor): 输入特征 [B, N, C]
            mask (Tensor, optional): 掩码 [B, N]。True 表示有效，False 表示 Padding。
        """
        # 投影
        q, k, v = (self.to_q(x), self.to_k(x), self.to_v(x))

        # 提取维度信息
        b, t, e, h, dh = *q.shape, self.heads, self.d_heads

        # Reshape & Transpose: [B, N, H*D] -> [B, N, H, D] -> [B, H, N, D]
        merge_heads = lambda x: x.reshape(b, t, h, dh).transpose(1, 2)
        q, k, v = map(merge_heads, (q, k, v))

        out = []

        split_index_fn = partial(split_at_index, 1, 0)

        (lq, q), (lk, k), (lv, v) = map(split_index_fn, (q, k, v))

        _, has_global = map(lambda x: x.shape[1] > 0, (lq, q))

        if has_global:
            global_out = self.global_attn_fn(q, k, v, kv_mask=mask)
            
            if self.custom_scale is not None:
                scale_correction = self.custom_scale / self.default_scale
                global_out = global_out * scale_correction
                
            out.append(global_out)

        # 拼接
        attn = torch.cat(out, dim=1)
        # Reshape back: [B, H, N, D] -> [B, N, H*D]
        attn = attn.transpose(1, 2).reshape(b, t, -1)
        
        return self.dropout(self.to_out(attn))