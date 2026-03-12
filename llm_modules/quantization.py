"""Quantization 量化模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Int8Linear(nn.Module):
    """INT8量化线性层"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randint(-128, 127, (out_dim, in_dim), dtype=torch.int8))
        self.scale = nn.Parameter(torch.ones(out_dim))

    def forward(self, x):
        w = self.weight.float() * self.scale.unsqueeze(1) / 127.0
        return F.linear(x, w)


class BinaryLinear(nn.Module):
    """二值化线性层"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        self.scale = nn.Parameter(torch.ones(out_dim))

    def forward(self, x):
        w_bin = torch.sign(self.weight) * self.scale.unsqueeze(1)
        return F.linear(x, w_bin)


def test_quantization():
    print("\n[测试] Quantization")
    x = torch.randn(2, 10, 128)

    int8 = Int8Linear(128, 128)
    out = int8(x)
    assert out.shape == (2, 10, 128)

    binary = BinaryLinear(128, 128)
    out = binary(x)
    assert out.shape == (2, 10, 128)
    print("✓ Int8Linear & BinaryLinear")


if __name__ == "__main__":
    test_quantization()
