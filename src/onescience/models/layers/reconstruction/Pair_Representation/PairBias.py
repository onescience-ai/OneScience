import torch
import torch.nn as nn
import torch.nn.functional as F

from onescience.registry import Pair_Representation

@Pair_Representation.registry_module()
class PairBias(nn.Module):
    """
    Pairwise Attention Bias
    ============================================================
    功能：
        将残基对 (i, j) 特征映射为注意力 bias，用于 MultiheadAttention。
        通常输入来自 PairEmbedding（几何、共变特征），输出形状匹配 [B, H, L, L]。

    输入：
        pair_repr: [B, L, L, C]  残基对表征（如距离、角度、MSA共变）
    
    输出：
        attn_bias: [B, H, L, L]  每个head的注意力偏置

    参数：
        pair_dim (int): 输入通道数
        num_heads (int): 注意力头数
        bias_dim (int): 每个head的bias特征维度（通常=C//num_heads）
        activation: 激活函数 (默认relu)
    """

    def __init__(
        self,
        pair_dim: int = 256,
        num_heads: int = 8,
        bias_dim: int = 32,
        activation: str = "relu",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.bias_dim = bias_dim

        # 投影层：将pair表征映射为每个head的bias
        self.linear = nn.Linear(pair_dim, num_heads * bias_dim)

        # 激活函数
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # 汇聚层（将bias_dim压缩为1）
        self.reduce = nn.Linear(bias_dim, 1, bias=False)

        # LayerNorm提升稳定性
        self.norm = nn.LayerNorm(pair_dim)

    def forward(self, pair_repr):
        """
        输入:
            pair_repr: [B, L, L, C]
        输出:
            attn_bias: [B, H, L, L]
        """
        B, L, _, C = pair_repr.shape

        # 归一化 + 线性映射
        x = self.norm(pair_repr)
        x = self.linear(x)  # [B, L, L, H*bias_dim]
        x = self.act(x)

        # reshape为多头格式
        x = x.view(B, L, L, self.num_heads, self.bias_dim)

        # 汇聚为每个head一个bias
        x = self.reduce(x).squeeze(-1)  # [B, L, L, H]
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, H, L, L]

        return x

if __name__ == "__main__":
    model = PairBias(pair_dim=256, num_heads=8, bias_dim=32)
    pair_repr = torch.randn(2, 128, 128, 256)
    out = model(pair_repr)
    print("attn_bias shape:", out.shape)  # [2, 8, 128, 128]
