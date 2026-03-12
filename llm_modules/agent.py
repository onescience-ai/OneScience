"""Agent 智能体模块"""

import torch.nn as nn
from .attention import MultiHeadAttention
from .normalization import RMSNorm


class ToolCallingHead(nn.Module):
    """工具调用头"""
    def __init__(self, dim, num_tools):
        super().__init__()
        self.tool_selector = nn.Linear(dim, num_tools)
        self.param_gen = nn.Linear(dim, dim)

    def forward(self, x):
        return self.tool_selector(x[:, -1]), self.param_gen(x[:, -1])


class ReflectionLayer(nn.Module):
    """反思层"""
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = MultiHeadAttention(dim, heads)
        self.critic = nn.Linear(dim, 1)
        self.norm = RMSNorm(dim)

    def forward(self, x):
        refined = x + self.attn(self.norm(x))
        confidence = torch.sigmoid(self.critic(refined[:, -1]))
        return refined, confidence


def test_agent():
    import torch
    print("\n[测试] Agent")
    x = torch.randn(2, 10, 128)

    tool_head = ToolCallingHead(128, num_tools=5)
    tool_logits, params = tool_head(x)
    assert tool_logits.shape == (2, 5)
    assert params.shape == (2, 128)

    reflect = ReflectionLayer(128, 4)
    refined, confidence = reflect(x)
    assert refined.shape == (2, 10, 128)
    assert confidence.shape == (2,)
    print("✓ ToolCallingHead & ReflectionLayer")


if __name__ == "__main__":
    test_agent()
