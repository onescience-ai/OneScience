import torch
import torch.nn as nn
import torch.nn.functional as F

from onescience.registry import Embedding

@Embedding.registry_module()
class MSAEmbedding(nn.Module):
    """
    多序列比对（MSA）嵌入层
    =================================================
    功能：
        - 将多条同源蛋白序列编码为高维表征；
        - 支持位置嵌入、层归一化、Dropout；
        - 可选地融合 query 序列信息（MSA首行）。

    输入：
        msa_seq: [B, N, L]   # B=batch, N=MSA条数, L=序列长度
        mask: [B, N, L]      # 掩码（1=有效, 0=padding）
        query_seq: [B, L] or None  # 可选的参考序列（中心序列）

    输出：
        msa_emb: [B, N, L, dim]  # 多序列嵌入表征

    参数：
        vocab_size (int): 氨基酸字典大小 (通常21)
        dim (int): 嵌入维度
        max_len (int): 序列最大长度
        dropout (float): dropout概率
        use_query_fusion (bool): 是否融合query序列
    """

    def __init__(
        self,
        vocab_size: int = 21,
        dim: int = 256,
        max_len: int = 1024,
        dropout: float = 0.1,
        use_query_fusion: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.use_query_fusion = use_query_fusion

        # MSA token embedding
        self.msa_embed = nn.Embedding(vocab_size, dim, padding_idx=0)

        # 可学习的位置编码
        self.pos_embed = nn.Embedding(max_len, dim)

        # query序列融合
        if use_query_fusion:
            self.query_proj = nn.Linear(dim, dim)

        # 归一化与dropout
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, msa_seq, mask=None, query_seq=None):
        """
        输入:
            msa_seq: [B, N, L]
            mask: [B, N, L]
            query_seq: [B, L] (optional)
        输出:
            msa_emb: [B, N, L, dim]
        """
        B, N, L = msa_seq.shape
        device = msa_seq.device

        # 基础token + position embedding
        msa_emb = self.msa_embed(msa_seq)
        pos = torch.arange(L, device=device).unsqueeze(0).unsqueeze(0).expand(B, N, L)
        msa_emb = msa_emb + self.pos_embed(pos)

        #  query序列融合（将第一条序列作为参考）
        if self.use_query_fusion and query_seq is not None:
            query_emb = self.msa_embed(query_seq)  # [B, L, dim]
            query_emb = self.query_proj(query_emb)
            msa_emb = msa_emb + query_emb.unsqueeze(1)  # 广播融合

        #  mask
        if mask is not None:
            msa_emb = msa_emb * mask.unsqueeze(-1)

        # 归一化 + dropout
        msa_emb = self.norm(msa_emb)
        msa_emb = self.dropout(msa_emb)
        return msa_emb


# ========== 使用示例 ==========
if __name__ == "__main__":
    model = MSAEmbedding(vocab_size=21, dim=256, max_len=512)
    msa_seq = torch.randint(0, 21, (2, 32, 128))  # batch=2, msa=32条序列, len=128
    mask = torch.ones_like(msa_seq)
    query_seq = msa_seq[:, 0]  # 第一条序列作为中心query
    out = model(msa_seq, mask, query_seq)
    print("msa_emb shape:", out.shape)  # [2, 32, 128, 256]
