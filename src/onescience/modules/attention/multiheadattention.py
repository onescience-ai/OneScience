import torch
import torch.nn as nn
from einops import rearrange

class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制 (Multi-Head Self-Attention) 。

    实现了标准的点积注意力：Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V。
    此实现保留了原始代码的特殊结构：即在通过线性层之前先将输入 Reshape 为多头格式。
    这通常意味着输入 Linear 层的权重矩阵是分头独立的 (H, D_head, D_head)。

    Args:
        dim (int): 输入特征维度 (注意：此参数在当前特定实现中仅用于 to_out 的输出维度，
                   输入投影层的维度由 heads * dim_head 隐式决定)。
        heads (int, optional): 注意力头数。默认值: 8。
        dim_head (int, optional): 每个注意力头的维度。默认值: 64。
        dropout (float, optional): Dropout 概率。默认值: 0.0。
        scale (float, optional): 自定义缩放因子。如果为 None，则使用 dim_head ** -0.5。默认值: None。
        is_causal (bool, optional): 是否应用因果掩码（Causal Masking），用于自回归任务。默认值: False。

    形状:
        输入 x: (B, N, C)，其中 C 必须等于 heads * dim_head (基于原代码逻辑)。
        输入 mask (可选): (B, N) 或 (B, 1, 1, N) 的布尔掩码。
        输出: (B, N, dim)。

    Example:
        >>> attn = MultiHeadAttention(dim=128, heads=8, dim_head=16)
        >>> # 注意：输入维度必须匹配 heads * dim_head = 8 * 16 = 128
        >>> x = torch.randn(8, 100, 128)
        >>> out = attn(x)
        >>> out.shape
        torch.Size([8, 100, 128])
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
        self.is_causal = is_causal
        
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        """
        前向传播。
        Args:
            x (Tensor): 输入张量 [B, N, C]
            mask (Tensor, optional): 注意力掩码。True 表示保留，False 表示屏蔽 (或相反，取决于具体实现习惯，这里通常处理为加性掩码)。
        """
        # B N C
        B, N, C = x.shape
        
        # x: [B, N, C] -> [B, N, H, D] -> [B, H, N, D]
        x = x.reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous() 
        
        # 投影
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        
        # 点积注意力
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        # 处理掩码
        if mask is not None or self.is_causal:
            mask_value = -torch.finfo(dots.dtype).max

            # 因果掩码
            if self.is_causal:
                i, j = dots.shape[-2:]
                causal_mask = torch.ones(i, j, device=x.device).triu(j - i + 1).bool()
                dots.masked_fill_(causal_mask, mask_value)

            if mask is not None:
                if mask.ndim == 2:
                    mask = mask.unsqueeze(1).unsqueeze(1)
                dots.masked_fill_(~mask.bool(), mask_value)

        attn = self.softmax(dots)
        attn = self.dropout(attn)
        
        # 加权求和
        res = torch.matmul(attn, v)  # B H N D
        
        # 拼接多头
        res = rearrange(res, 'b h n d -> b n (h d)')
        
        return self.to_out(res)