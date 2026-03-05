import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class FlashAttention(nn.Module):
    """
    FlashAttention 模块。

    利用 PyTorch 的 F.scaled_dot_product_attention 实现的高效注意力机制。
    该实现完全兼容 PyTorch 2.0+ 的 SDPA 加速（FlashAttention V2, Memory-Efficient Attention, C++ Math）。
    它通过 Tiling 技术减少 HBM (高带宽内存) 访问次数，显著提升长序列的训练和推理速度。

    Args:
        dim (int): 输入特征维度。
        heads (int, optional): 注意力头数。默认值: 8。
        dim_head (int, optional): 每个注意力头的维度。默认值: 64。
        dropout (float, optional): Dropout 概率。默认值: 0.0。
        scale (float, optional): 自定义缩放因子。如果为 None，则使用 dim_head ** -0.5。默认值: None。
        is_causal (bool, optional): 是否应用因果掩码（Causal Masking），用于自回归任务。默认值: False。

    形状:
        输入 x: (B, N, C)，其中 B 是批次大小，N 是序列长度，C 是特征维度。
        输入 mask (可选): (B, N) 或 (B, 1, 1, N) 的布尔/浮点掩码。
        输出: (B, N, C)。

    Example:
        >>> attn = FlashAttention(dim=64, heads=8, dim_head=8)
        >>> x = torch.randn(8, 128, 64)
        >>> out = attn(x)
        >>> print(out.shape)
        torch.Size([8, 128, 64])
    """
    def __init__(
        self, 
        dim, 
        heads=8, 
        dim_head=64, 
        dropout=0., 
        scale=None, 
        is_causal=False, 
        **kwargs
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = scale if scale is not None else dim_head ** -0.5
        self.dropout_p = dropout 
        self.is_causal = is_causal
        
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        """
        前向传播。
        
        Args:
            x (Tensor): 输入张量 [B, N, C]
            mask (Tensor, optional): 注意力掩码。
        """
        # x shape: [batch_size, seq_len, dim]
        batch_size, seq_len, _ = x.shape
        
        # Get query, key, value projections
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        
        # Reshape for multi-head attention: [B, N, H*D] -> [B, H, N, D]
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads).contiguous()
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads).contiguous()
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads).contiguous()
        
        # 处理 Mask
        if mask is not None:
            if mask.ndim == 2: # [B, N] -> [B, 1, 1, N]
                mask = mask.unsqueeze(1).unsqueeze(1)
            pass 

        # Flash attention implementation
        attn_output = F.scaled_dot_product_attention(
            query=q, 
            key=k, 
            value=v,
            attn_mask=mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=self.is_causal,
            scale=self.scale
        )

        # Reshape back: [B, H, N, D] -> [B, N, H*D]
        out = rearrange(attn_output, 'b h n d -> b n (h d)')    
        return self.to_out(out)