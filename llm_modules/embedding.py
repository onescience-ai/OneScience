"""Embedding 嵌入层模块"""

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """词嵌入层"""
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)

    def forward(self, x):
        return self.emb(x)


class RoPE(nn.Module):
    """旋转位置编码 (Rotary Position Embedding)"""
    def __init__(self, dim, max_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer('cos', freqs.cos())
        self.register_buffer('sin', freqs.sin())

    def forward(self, x):
        seq_len = x.shape[-2]
        cos = self.cos[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin[:seq_len].unsqueeze(0).unsqueeze(0)
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class ALiBi(nn.Module):
    """注意力线性偏置 (Attention with Linear Biases)"""
    def __init__(self, heads):
        super().__init__()
        slopes = torch.pow(2, -torch.arange(1, heads + 1) * 8.0 / heads)
        self.register_buffer('slopes', slopes.view(1, heads, 1, 1))

    def forward(self, seq_len):
        pos = torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)
        return pos.unsqueeze(0).unsqueeze(0) * self.slopes


class LearnedPositionalEmbedding(nn.Module):
    """可学习位置编码"""
    def __init__(self, max_len, dim):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, dim)

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device)
        return x + self.pos_emb(pos)


def test_embedding():
    print("\n[测试] Embedding")
    emb = TokenEmbedding(1000, 128)
    tokens = torch.randint(0, 1000, (2, 10))
    out = emb(tokens)
    assert out.shape == (2, 10, 128)

    rope = RoPE(64)
    x = torch.randn(2, 4, 10, 64)
    out = rope(x)

    alibi = ALiBi(4)
    bias = alibi(10)
    assert bias.shape == (1, 4, 10, 10)

    learned_pe = LearnedPositionalEmbedding(512, 128)
    x = torch.randn(2, 10, 128)
    out = learned_pe(x)
    assert out.shape == (2, 10, 128)
    print("✓ TokenEmbedding, RoPE, ALiBi & LearnedPE")


if __name__ == "__main__":
    test_embedding()
