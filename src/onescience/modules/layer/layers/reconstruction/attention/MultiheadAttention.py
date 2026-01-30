import torch
import torch.nn as nn
import torch.nn.functional as F

from onescience.registry import Attention

@Attention.registry_module()
class MultiheadAttention(nn.Module):
    """
    Multi-head Scaled Dot-Product Attention
    ============================================================
    通用多头注意力层 (兼容 Protenix / OpenFold / Evo2)
    
    功能：
        - 支持可选的 attention bias（如pair信息、模板约束）
        - 支持 mask
        - 可选 residual、dropout
        - 可扩展：cross-attention / self-attention 通用实现

    输入:
        query: [B, L, dim]
        key:   [B, L, dim] 或 [B, S, dim]
        value: [B, L, dim] 或 [B, S, dim]
        mask:  [B, 1, L, S] (可选)
        attn_bias: [B, num_heads, L, S] (可选)
    
    输出:
        out: [B, L, dim]
    """

    def __init__(self, dim, num_heads=8, dropout=0.1, bias=True):
        super().__init__()
        assert dim % num_heads == 0, "dim必须能整除num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 线性投影层
        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.o_proj = nn.Linear(dim, dim, bias=bias)

        # Dropout
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, query, key=None, value=None, mask=None, attn_bias=None):
        """
        执行注意力计算
        """
        if key is None:
            key = query
        if value is None:
            value = query

        B, L, _ = query.shape
        S = key.shape[1]  # key长度

        # 线性映射 + 分多头
        q = self.q_proj(query).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, L, d]
        k = self.k_proj(key).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)    # [B, h, S, d]
        v = self.v_proj(value).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, S, d]

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, h, L, S]

        # 加入bias（如pair bias或几何约束）
        if attn_bias is not None:
            attn_scores = attn_scores + attn_bias

        # mask无效区域
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # softmax
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_drop(attn_probs)

        # 加权求和
        out = torch.matmul(attn_probs, v)  # [B, h, L, d]
        out = out.transpose(1, 2).contiguous().view(B, L, self.dim)

        # 输出投影
        out = self.o_proj(out)
        out = self.proj_drop(out)

        return out
        
# ========== 使用示例 ==========
if __name__ == "__main__":
    attn = MultiheadAttention(dim=256, num_heads=8)
    q = torch.randn(2, 128, 256)
    mask = torch.ones(2, 1, 128, 128)
    out = attn(q, mask=mask)
    print("output shape:", out.shape)  # [2, 128, 256]
