import torch
import torch.nn as nn
import torch.nn.functional as F

from onescience.registry import Attention

@Attention.registry_module()
class TriangleAttention(nn.Module):
    """
    TriangleAttention Layer
    =====================================================
    功能:
        建模残基三元 (i, j, k) 的几何依赖关系；
        是 AlphaFold2 / OpenFold / Protenix / Evo2 中的核心几何注意力层。
    
    两种方向:
        1️⃣ per_row     —— TriangleAttentionStartingNode
        2️⃣ per_column  —— TriangleAttentionEndingNode

    输入:
        pair_repr: [B, L, L, C] 残基对表征
        mask: [B, L, L] (optional)

    输出:
        updated_pair: [B, L, L, C]
    """

    def __init__(
        self,
        pair_dim: int = 256,
        num_heads: int = 8,
        orientation: str = "per_row",  # per_row 或 per_column
        dropout: float = 0.1,
    ):
        super().__init__()
        assert orientation in ["per_row", "per_column"], "orientation must be 'per_row' or 'per_column'"
        self.orientation = orientation
        self.pair_dim = pair_dim
        self.num_heads = num_heads
        self.head_dim = pair_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # QKV projection
        self.q_proj = nn.Linear(pair_dim, pair_dim)
        self.k_proj = nn.Linear(pair_dim, pair_dim)
        self.v_proj = nn.Linear(pair_dim, pair_dim)
        self.o_proj = nn.Linear(pair_dim, pair_dim)

        # Normalization + Dropout
        self.layer_norm = nn.LayerNorm(pair_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, pair_repr, mask=None):
        """
        pair_repr: [B, L, L, C]
        mask: [B, L, L] (可选)  — padding 掩码
        """
        B, L, _, C = pair_repr.shape
        x = self.layer_norm(pair_repr)

        # QKV projection
        q = self.q_proj(x).view(B, L, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, L, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, L, self.num_heads, self.head_dim)

        # --- 方向决定注意力维度 ---
        # per_row: 固定 i，沿 (j,k) 计算注意力
        # per_column: 固定 j，沿 (i,k) 计算注意力
        if self.orientation == "per_row":
            q = q.permute(0, 2, 3, 1, 4)  # [B, L_j, H, L_i, D]
            k = k.permute(0, 2, 3, 1, 4)
            v = v.permute(0, 2, 3, 1, 4)
        else:
            q = q.permute(0, 1, 3, 2, 4)  # [B, L_i, H, L_j, D]
            k = k.permute(0, 1, 3, 2, 4)
            v = v.permute(0, 1, 3, 2, 4)

        # --- 计算注意力 ---
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [..., L, L]

        # 掩码处理
        if mask is not None:
            if self.orientation == "per_row":
                mask_ = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L, L]
            else:
                mask_ = mask.unsqueeze(1).unsqueeze(2)
            attn_logits = attn_logits.masked_fill(mask_ == 0, float("-inf"))

        attn_probs = F.softmax(attn_logits, dim=-1)
        attn_out = torch.matmul(attn_probs, v)

        # --- 恢复形状 ---
        if self.orientation == "per_row":
            attn_out = attn_out.permute(0, 3, 1, 2, 4).contiguous().view(B, L, L, C)
        else:
            attn_out = attn_out.permute(0, 1, 3, 2, 4).contiguous().view(B, L, L, C)

        out = self.o_proj(attn_out)
        out = self.dropout(out)

        # 残差连接
        updated_pair = pair_repr + out

        return updated_pair


if __name__ == "__main__":
    B, L, C = 1, 64, 256
    pair = torch.randn(B, L, L, C)

    tri_row = TriangleAttention(pair_dim=C, orientation="per_row")
    tri_col = TriangleAttention(pair_dim=C, orientation="per_column")

    out_row = tri_row(pair)
    out_col = tri_col(pair)

    print("Row:", out_row.shape)  # [1, 64, 64, 256]
    print("Col:", out_col.shape)
