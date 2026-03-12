"""MoE 混合专家模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .feedforward import SwiGLU


class MoE(nn.Module):
    """混合专家层"""
    def __init__(self, dim, hidden_dim, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(dim, hidden_dim) for _ in range(num_experts)])

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(-1, C)
        gate_logits = self.gate(x_flat)
        weights, indices = torch.topk(gate_logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        out = torch.zeros_like(x_flat)
        for i in range(self.top_k):
            expert_idx = indices[:, i]
            expert_weight = weights[:, i:i+1]
            for e in range(self.num_experts):
                mask = expert_idx == e
                if mask.any():
                    out[mask] += expert_weight[mask] * self.experts[e](x_flat[mask])
        return out.view(B, T, C)


class MixtureOfAdapters(nn.Module):
    """混合适配器"""
    def __init__(self, dim, num_adapters=4, bottleneck=64):
        super().__init__()
        self.gate = nn.Linear(dim, num_adapters)
        from .peft import Adapter
        self.adapters = nn.ModuleList([Adapter(dim, bottleneck) for _ in range(num_adapters)])

    def forward(self, x):
        weights = F.softmax(self.gate(x.mean(1)), dim=-1)
        out = sum(w.view(-1, 1, 1) * adapter(x) for w, adapter in zip(weights.T, self.adapters))
        return out


class RAMoLE(nn.Module):
    """检索增强混合专家"""
    def __init__(self, dim, hidden_dim, num_experts=4):
        super().__init__()
        self.retriever = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([SwiGLU(dim, hidden_dim) for _ in range(num_experts)])
        self.fusion = nn.Linear(dim * 2, dim)

    def forward(self, x, memory_bank):
        scores = torch.softmax(self.retriever(x), dim=-1)
        expert_out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            expert_out += scores[..., i:i+1] * expert(x)
        retrieved = torch.matmul(scores, memory_bank)
        fused = torch.cat([expert_out, retrieved], dim=-1)
        gate = torch.sigmoid(self.fusion(fused))
        return gate * expert_out + (1 - gate) * retrieved


def test_moe():
    print("\n[测试] MoE")
    x = torch.randn(2, 10, 128)

    moe = MoE(128, 512, num_experts=4, top_k=2)
    out = moe(x)
    assert out.shape == (2, 10, 128)

    moa = MixtureOfAdapters(128, num_adapters=4, bottleneck=32)
    out = moa(x)
    assert out.shape == (2, 10, 128)

    memory = torch.randn(4, 128)
    ramole = RAMoLE(128, 512, num_experts=4)
    out = ramole(x, memory)
    assert out.shape == (2, 10, 128)
    print("✓ MoE, MixtureOfAdapters & RAMoLE")


if __name__ == "__main__":
    test_moe()
