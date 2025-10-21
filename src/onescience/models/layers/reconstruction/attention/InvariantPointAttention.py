import torch
import torch.nn as nn
import torch.nn.functional as F

from onescience.registry import Attention

@Attention.registry_module()
class TriangleAttention(nn.Module):
    """
    Triangle Attention Module
    ==============================================================
    功能:
        - 在残基对 (i, j) 表征中，引入第三个中介残基 k；
        - 用注意力机制建模三元几何关系；
        - 分为两个方向：
            (1) "starting node" 方向 (沿 i->k)
            (2) "ending node"   方向 (沿 j->k)
        - 被广泛用于 AlphaFold2、OpenFold、Protenix、Evo2 的几何 trunk。

    输入:
        pair_repr: [B, L, L, C] 残基对特征 (对称矩阵)
        mask:      [B, L, L] (可选) padding 掩码

    输出:
        updated_pair: [B, L, L, C]
    """

    def __init__(
        self,
        pair_dim: int = 256,
        num_heads: int = 8,
        orientation: str = "per_row",  # or "per_column"
        dropout: float = 0.1,
    ):
        super().__init__()
        assert orientation in ["per_row", "per_column"], "orientation must be per_row or per_column"

        self.pair_dim = pair_dim
        self.num_heads = num_heads
        self.head_dim = pair_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.orientation = orientation

        # QKV projections
        self.q_proj = nn.Linear(pair_dim, pair_dim)
        self.k_proj = nn.Linear(pair_dim, pair_dim)
        self.v_proj = nn.Linear(pair_dim, pair_dim)
        self.o_proj = nn.Linear(pair_dim, pair_dim)

        # Normalization + dropout
        self.norm = nn.LayerNorm(pair_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, pair_repr, mask=None):
        """
        pair_repr: [B, L, L, C]
        mask: [B, L, L] (optional)
        """
        B, L, _, C = pair_repr.shape

        x = self.norm(pair_repr)

        # Q,K,V 投影
        q = self.q_proj(x).view(B, L, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, L, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, L, self.num_heads, self.head_dim)

        if self.orientation == "per_row":
            # 每一行 (i 固定) 看成序列，对 (j,k) 计算注意力
            q = q.permute(0, 2, 3, 1, 4)  # [B, L_j, H, L_i, D]
            k = k.permute(0, 2, 3, 1, 4)
            v = v.permute(0, 2, 3, 1, 4)

        elif self.orientation == "per_column":
            # 每一列 (j 固定)
            q = q.permute(0, 1, 3, 2, 4)
            k = k.permute(0, 1, 3, 2, 4)
            v = v.permute(0, 1, 3, 2, 4)

        # 点积注意力
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [..., L, L]

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask[:, None, None, :, :] == 0, float("-inf"))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_probs, v)

        # reshape 回原状
        if self.orientation == "per_row":
            attn_out = attn_out.permute(0, 3, 1, 2, 4).contiguous().view(B, L, L, C)
        else:
            attn_out = attn_out.permute(0, 1, 3, 2, 4).contiguous().view(B, L, L, C)

        out = self.o_proj(attn_out)
        pair_repr = pair_repr + self.dropout(out)

        return pair_repr

if __name__ == "__main__":
    model_row = TriangleAttention(pair_dim=256, orientation="per_row")
    model_col = TriangleAttention(pair_dim=256, orientation="per_column")

    pair = torch.randn(1, 128, 128, 256)
    out1 = model_row(pair)
    out2 = model_col(pair)

    print(out1.shape, out2.shape)  # (1, 128, 128, 256)
