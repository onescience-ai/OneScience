import torch
import torch.nn as nn
import torch.nn.functional as F

from onescience.registry import Embedding

@Embedding.registry_module()
class TemplateEmbedding(nn.Module):
    """
    Template Embedding Layer
    ============================================================
    功能：
        - 将模板的氨基酸序列、原子距离、角度等特征编码成高维表征；
        - 支持序列嵌入、几何距离编码、模板对齐 mask；
        - 输出残基对（pairwise）表征，供 Transformer 主干使用。

    输入：
        template_seq: [B, T, L]   模板序列 (T个模板)
        template_dist: [B, T, L, L]  残基-残基距离矩阵
        template_mask: [B, T, L, L]  有效掩码

    输出：
        template_pair_emb: [B, L, L, dim]  模板对嵌入特征

    参数：
        vocab_size (int): 氨基酸字典大小
        dim (int): 嵌入维度
        max_dist (float): 距离截断阈值（Å）
        num_bins (int): 距离量化区间数
        dropout (float): dropout概率
    """

    def __init__(
        self,
        vocab_size: int = 21,
        dim: int = 256,
        max_dist: float = 20.0,
        num_bins: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_bins = num_bins
        self.max_dist = max_dist

        # 模板序列 token embedding
        self.seq_embed = nn.Embedding(vocab_size, dim, padding_idx=0)

        # 距离嵌入（通过 binning + embedding）
        self.dist_bins = torch.linspace(0, max_dist, num_bins - 1)
        self.dist_embed = nn.Embedding(num_bins, dim)

        # 3模板 pair feature 融合层
        self.pair_linear = nn.Linear(2 * dim + dim, dim)

        # 正则层
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def _discretize_distances(self, d):
        """
        将距离连续值映射为离散bin索引
        """
        d = torch.clamp(d, 0, self.max_dist)
        idx = torch.bucketize(d, self.dist_bins, right=True)
        return idx

    def forward(self, template_seq, template_dist, template_mask=None):
        """
        输入:
            template_seq: [B, T, L]
            template_dist: [B, T, L, L]
            template_mask: [B, T, L, L]
        输出:
            template_pair_emb: [B, L, L, dim]
        """
        B, T, L = template_seq.shape
        device = template_seq.device

        # 序列嵌入
        seq_emb = self.seq_embed(template_seq)  # [B, T, L, dim]

        # 距离嵌入（量化+embedding）
        dist_idx = self._discretize_distances(template_dist)
        dist_emb = self.dist_embed(dist_idx)  # [B, T, L, L, dim]

        # 生成 pair feature
        seq_i = seq_emb.unsqueeze(3).expand(B, T, L, L, self.dim)
        seq_j = seq_emb.unsqueeze(2).expand(B, T, L, L, self.dim)

        pair_emb = torch.cat([seq_i, seq_j, dist_emb], dim=-1)  # [B, T, L, L, 3*dim]
        pair_emb = self.pair_linear(pair_emb)

        # 模板间聚合 (平均)
        pair_emb = pair_emb.mean(dim=1)  # [B, L, L, dim]

        # 掩码、归一化、dropout
        if template_mask is not None:
            pair_emb = pair_emb * template_mask[:, 0, :, :, None]

        pair_emb = self.norm(pair_emb)
        pair_emb = self.dropout(pair_emb)

        return pair_emb


# ========== 使用示例 ==========
if __name__ == "__main__":
    B, T, L = 2, 4, 128
    model = TemplateEmbedding(vocab_size=21, dim=256)
    template_seq = torch.randint(0, 21, (B, T, L))
    template_dist = torch.rand(B, T, L, L) * 20
    template_mask = torch.ones(B, T, L, L)
    out = model(template_seq, template_dist, template_mask)
    print("template_pair_emb shape:", out.shape)  # [2, 128, 128, 256]
