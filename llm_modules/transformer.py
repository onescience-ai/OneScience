"""Transformer 模块"""

import torch.nn as nn
from .attention import MultiHeadAttention
from .feedforward import SwiGLU
from .normalization import RMSNorm


class TransformerBlock(nn.Module):
    """标准Transformer块"""
    def __init__(self, dim, heads, hidden_dim, dropout=0.1):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, heads, dropout)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, hidden_dim)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


def test_transformer():
    import torch
    print("\n[测试] Transformer")
    x = torch.randn(2, 10, 128)
    block = TransformerBlock(128, heads=4, hidden_dim=512)
    out = block(x)
    assert out.shape == (2, 10, 128)
    print("✓ TransformerBlock")


if __name__ == "__main__":
    test_transformer()
