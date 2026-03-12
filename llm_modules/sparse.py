"""Sparse 稀疏门控模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKGating(nn.Module):
    """Top-K门控"""
    def __init__(self, dim, num_experts, top_k=2):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts)
        self.top_k = top_k

    def forward(self, x):
        logits = self.gate(x)
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(top_k_logits, dim=-1)
        return weights, top_k_indices


class ExpertChoice(nn.Module):
    """专家选择路由"""
    def __init__(self, dim, num_experts, capacity_factor=1.0):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts)
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor

    def forward(self, x):
        B, T, C = x.shape
        logits = self.gate(x.view(-1, C))
        probs = F.softmax(logits, dim=-1)
        capacity = int(B * T * self.capacity_factor / self.num_experts)
        expert_indices = torch.argmax(probs, dim=-1)
        return expert_indices, capacity


def test_sparse():
    print("\n[测试] Sparse")
    x = torch.randn(2, 10, 128)

    topk_gate = TopKGating(128, num_experts=4, top_k=2)
    weights, indices = topk_gate(x)
    assert weights.shape == (2, 10, 2)

    expert_choice = ExpertChoice(128, num_experts=4)
    indices, capacity = expert_choice(x)
    assert indices.shape == (20,)
    print("✓ TopKGating & ExpertChoice")


if __name__ == "__main__":
    test_sparse()
