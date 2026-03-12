"""Feedforward 前馈网络模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU激活函数"""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class GeGLU(nn.Module):
    """GeGLU激活函数"""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.proj(F.gelu(self.w1(x)) * self.w2(x))


class GLU(nn.Module):
    """门控线性单元 (Gated Linear Unit)"""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.proj(torch.sigmoid(self.w1(x)) * self.w2(x))


class Mish(nn.Module):
    """Mish激活函数"""
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


def test_feedforward():
    print("\n[测试] Feedforward")
    x = torch.randn(2, 10, 128)

    swiglu = SwiGLU(128, 512)
    out = swiglu(x)
    assert out.shape == (2, 10, 128)

    geglu = GeGLU(128, 512)
    out = geglu(x)
    assert out.shape == (2, 10, 128)

    glu = GLU(128, 512)
    out = glu(x)
    assert out.shape == (2, 10, 128)

    mish = Mish()
    out = mish(x)
    assert out.shape == (2, 10, 128)
    print("✓ SwiGLU, GeGLU, GLU & Mish")


if __name__ == "__main__":
    test_feedforward()
