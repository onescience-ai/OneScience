"""RAG 知识增强模块"""

import torch.nn as nn
import torch.nn.functional as F
from .attention import MultiHeadAttention
from .feedforward import SwiGLU
from .normalization import RMSNorm


class CrossAttention(nn.Module):
    """交叉注意力"""
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, dim * 2, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, context):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        kv = self.kv(context).view(B, -1, 2, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class RAGLayer(nn.Module):
    """RAG检索增强生成层"""
    def __init__(self, dim, heads):
        super().__init__()
        self.self_attn = MultiHeadAttention(dim, heads)
        self.cross_attn = CrossAttention(dim, heads)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.norm3 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, dim * 4)

    def forward(self, x, retrieved_docs):
        x = x + self.self_attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), retrieved_docs)
        x = x + self.ffn(self.norm3(x))
        return x


def test_rag():
    import torch
    print("\n[测试] RAG")
    x = torch.randn(2, 10, 128)
    docs = torch.randn(2, 5, 128)

    cross_attn = CrossAttention(128, 4)
    out = cross_attn(x, docs)
    assert out.shape == (2, 10, 128)

    rag = RAGLayer(128, 4)
    out = rag(x, docs)
    assert out.shape == (2, 10, 128)
    print("✓ CrossAttention & RAGLayer")


if __name__ == "__main__":
    test_rag()
