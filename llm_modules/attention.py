"""Attention 注意力机制模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class GroupedQueryAttention(nn.Module):
    """分组查询注意力 (Grouped-Query Attention)"""
    def __init__(self, dim, heads, kv_heads):
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, 2 * kv_heads * self.head_dim, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        kv = self.kv(x).view(B, T, 2, self.kv_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        k = k.repeat_interleave(self.heads // self.kv_heads, dim=1)
        v = v.repeat_interleave(self.heads // self.kv_heads, dim=1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class SlidingWindowAttention(nn.Module):
    """滑动窗口注意力"""
    def __init__(self, dim, heads, window_size=512):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.ones(T, T, device=x.device).tril()
        for i in range(T):
            mask[i, :max(0, i - self.window_size)] = 0
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class FlashAttention(nn.Module):
    """Flash Attention"""
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(out.transpose(1, 2).reshape(B, T, C))


def test_attention():
    print("\n[测试] Attention")
    x = torch.randn(2, 10, 128)

    mha = MultiHeadAttention(128, 4)
    out = mha(x)
    assert out.shape == (2, 10, 128)

    gqa = GroupedQueryAttention(128, heads=4, kv_heads=2)
    out = gqa(x)
    assert out.shape == (2, 10, 128)

    swa = SlidingWindowAttention(128, 4, window_size=5)
    out = swa(x)
    assert out.shape == (2, 10, 128)

    flash = FlashAttention(128, 4)
    out = flash(x)
    assert out.shape == (2, 10, 128)
    print("✓ MHA, GQA, SlidingWindow & FlashAttention")


if __name__ == "__main__":
    test_attention()
