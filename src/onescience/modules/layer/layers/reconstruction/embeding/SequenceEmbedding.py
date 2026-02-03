# protein_model_base/embedding.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from onescience.registry import Embedding

@Embedding.registry_module()
class SequenceEmbedding(nn.Module):
    """
    序列特征嵌入层 (shared by OpenFold / Protenix / Evo2 / ESM)
    =============================================================

    功能：
        - 将氨基酸序列（或token序列）转换为连续表征；
        - 支持 residue embedding + positional encoding；
        - 可融合 one-hot / property / MSA 信息；
        - 支持 mask、dropout、LayerNorm。

    输入：
        seq: [B, L]  氨基酸ID序列 (0~vocab_size-1)
        mask: [B, L] 序列掩码 (1=有效, 0=padding)

    输出：
        emb: [B, L, dim]  序列嵌入表示

    参数：
        vocab_size (int): 氨基酸种类数，默认为 21 (20个AA + mask)
        dim (int): 嵌入维度
        max_len (int): 位置编码的最大长度
        dropout (float): dropout比率
        use_pos_encoding (bool): 是否启用位置编码
    """

    def __init__(
        self,
        vocab_size: int = 21,
        dim: int = 256,
        max_len: int = 1024,
        dropout: float = 0.1,
        use_pos_encoding: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_len = max_len
        self.use_pos_encoding = use_pos_encoding

        # 序列embedding（one-hot 替代）
        self.token_embed = nn.Embedding(vocab_size, dim, padding_idx=0)

        # 可学习的位置编码（可换成sinusoidal）
        if use_pos_encoding:
            self.pos_embed = nn.Embedding(max_len, dim)

        # LayerNorm + Dropout
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        # 额外氨基酸性质特征 (hydrophobic, charge, etc.)
        # 可选: 用于 Evo2 / Protenix 融合
        self.use_property = True
        if self.use_property:
            self.prop_embed = nn.Linear(8, dim)  # 8种生化属性特征占位

    def forward(self, seq, mask=None, aa_property=None):
        """
        参数：
            seq: [B, L] token ID
            mask: [B, L] padding掩码 (1有效, 0无效)
            aa_property: [B, L, 8] 可选的生化特征
        返回：
            x: [B, L, dim]
        """
        B, L = seq.shape
        device = seq.device

        # token embedding
        x = self.token_embed(seq)  # [B,L,dim]

        # 位置编码
        if self.use_pos_encoding:
            pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
            x = x + self.pos_embed(pos)

        # 属性特征
        if self.use_property and aa_property is not None:
            x = x + self.prop_embed(aa_property)

        # 掩码填充 (padding处置零)
        if mask is not None:
            x = x * mask.unsqueeze(-1)

        x = self.norm(x)
        x = self.dropout(x)
        return x


# ========== 使用示例 ==========
if __name__ == "__main__":
    model = SequenceEmbedding(vocab_size=21, dim=256, max_len=512)
    seq = torch.randint(0, 21, (2, 100))     # 2条蛋白序列，每条长度100
    mask = torch.ones_like(seq)               # 全部有效
    output = model(seq, mask)
    print("output shape:", output.shape)  # [2, 100, 256]
