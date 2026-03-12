"""Normalization 归一化模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """RMS归一化 (Root Mean Square Normalization)"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class LayerNorm(nn.Module):
    """层归一化 (Layer Normalization)"""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        return F.layer_norm(x, (x.size(-1),), self.weight, self.bias, self.eps)


class GroupNorm(nn.Module):
    """组归一化 (Group Normalization)"""
    def __init__(self, dim, num_groups=32):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, dim)

    def forward(self, x):
        B, T, C = x.shape
        return self.gn(x.transpose(1, 2)).transpose(1, 2)


def test_normalization():
    print("\n[测试] Normalization")
    x = torch.randn(2, 10, 128)

    rms = RMSNorm(128)
    out = rms(x)
    assert out.shape == (2, 10, 128)

    ln = LayerNorm(128)
    out = ln(x)
    assert out.shape == (2, 10, 128)

    gn = GroupNorm(128, num_groups=8)
    out = gn(x)
    assert out.shape == (2, 10, 128)
    print("✓ RMSNorm, LayerNorm & GroupNorm")


if __name__ == "__main__":
    test_normalization()
