import torch
import torch.nn as nn
import torch.nn.functional as F

from onescience.registry import Pair_Representation

@Pair_Representation.registry_module()
class PairUpdate(nn.Module):
    """
    Pair Representation Update Block
    ============================================================
    功能：
        - 对 pair (i,j) 表征执行 self-attention 和前馈更新；
        - 使用 PairBias 引导注意力；
        - 在 Protenix / OpenFold / Evo2 的 Evoformer trunk 中广泛使用。

    输入：
        pair_repr: [B, L, L, C]   残基对表征
        pair_bias: [B, H, L, L]   注意力偏置（来自 PairBias）

    输出：
        updated_pair: [B, L, L, C]
    """

    def __init__(
        self,
        pair_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        ff_hidden_dim: int = 512,
    ):
        super().__init__()
        self.pair_dim = pair_dim
        self.num_heads = num_heads
        self.head_dim = pair_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 注意力权重
        self.q_proj = nn.Linear(pair_dim, pair_dim)
        self.k_proj = nn.Linear(pair_dim, pair_dim)
        self.v_proj = nn.Linear(pair_dim, pair_dim)
        self.o_proj = nn.Linear(pair_dim, pair_dim)

        # 前馈层
        self.ff = nn.Sequential(
            nn.Linear(pair_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, pair_dim),
            nn.Dropout(dropout),
        )

        # 归一化
        self.norm1 = nn.LayerNorm(pair_dim)
        self.norm2 = nn.LayerNorm(pair_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, pair_repr, pair_bias=None, mask=None):
        """
        输入:
            pair_repr: [B, L, L, C]
            pair_bias: [B, H, L, L] (optional)
            mask: [B, L, L] (optional)
        输出:
            updated_pair: [B, L, L, C]
        """
        B, L, _, C = pair_repr.shape

        # Self-Attention 操作
        x = self.norm1(pair_repr)

        # reshape for attention: treat j dimension as sequence
        q = self.q_proj(x).view(B, L, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, L, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, L, self.num_heads, self.head_dim)

        # transpose for matmul: [B, L, H, L, D]
        q = q.permute(0, 2, 3, 1, 4)  # [B, L_j, H, L_i, D]
        k = k.permute(0, 2, 3, 4, 1)
        v = v.permute(0, 2, 3, 1, 4)

        # Attention along i-j pair axes
        attn_scores = torch.matmul(q, k) * self.scale  # [B, L, H, L, L]

        # 加入 bias
        if pair_bias is not None:
            attn_scores = attn_scores + pair_bias.unsqueeze(1)  # broadcast over L_j

        # Mask padding
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask[:, None, None, :, :] == 0, float("-inf"))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_probs, v)  # [B, L, H, L, D]
        attn_out = attn_out.permute(0, 3, 1, 2, 4).contiguous().view(B, L, L, C)

        attn_out = self.o_proj(attn_out)
        pair_repr = pair_repr + self.dropout(attn_out)  # 残差连接

        # 前馈网络更新
        ff_out = self.ff(self.norm2(pair_repr))
        updated_pair = pair_repr + ff_out  # 残差连接

        return updated_pair

if __name__ == "__main__":
    model = PairUpdate(pair_dim=256, num_heads=8)
    pair_repr = torch.randn(2, 128, 128, 256)
    pair_bias = torch.randn(2, 8, 128, 128)
    out = model(pair_repr, pair_bias)
    print("updated_pair shape:", out.shape)  # [2, 128, 128, 256]
