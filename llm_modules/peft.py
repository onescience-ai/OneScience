"""PEFT 参数高效微调模块"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """LoRA低秩适配"""
    def __init__(self, in_dim, out_dim, rank=8, alpha=16):
        super().__init__()
        self.base = nn.Linear(in_dim, out_dim, bias=False)
        self.lora_A = nn.Linear(in_dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_dim, bias=False)
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x)) * self.scale


class QLoRALinear(nn.Module):
    """QLoRA量化低秩适配"""
    def __init__(self, in_dim, out_dim, rank=8, bits=4):
        super().__init__()
        self.weight_quant = nn.Parameter(torch.randint(-2**(bits-1), 2**(bits-1), (out_dim, in_dim)))
        self.scale = nn.Parameter(torch.ones(out_dim))
        self.lora_A = nn.Linear(in_dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_dim, bias=False)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        w_dequant = self.weight_quant.float() * self.scale.unsqueeze(1)
        base_out = F.linear(x, w_dequant)
        return base_out + self.lora_B(self.lora_A(x))


class Adapter(nn.Module):
    """Adapter微调层"""
    def __init__(self, dim, bottleneck_dim=64):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, dim)
        self.act = nn.GELU()
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))


class PrefixTuning(nn.Module):
    """前缀微调"""
    def __init__(self, prefix_len, dim):
        super().__init__()
        self.prefix = nn.Parameter(torch.randn(prefix_len, dim))

    def forward(self, x):
        B = x.size(0)
        prefix = self.prefix.unsqueeze(0).expand(B, -1, -1)
        return torch.cat([prefix, x], dim=1)


class PromptTuning(nn.Module):
    """提示微调"""
    def __init__(self, num_prompts, dim):
        super().__init__()
        self.prompts = nn.Parameter(torch.randn(num_prompts, dim))

    def forward(self, x, prompt_ids):
        return x + self.prompts[prompt_ids]


def test_peft():
    print("\n[测试] PEFT")
    x = torch.randn(2, 10, 128)

    lora = LoRALinear(128, 128, rank=8)
    out = lora(x)
    assert out.shape == (2, 10, 128)

    qlora = QLoRALinear(128, 128, rank=8)
    out = qlora(x)
    assert out.shape == (2, 10, 128)

    adapter = Adapter(128, 32)
    out = adapter(x)
    assert out.shape == (2, 10, 128)

    prefix = PrefixTuning(5, 128)
    out = prefix(x)
    assert out.shape == (2, 15, 128)

    prompt = PromptTuning(10, 128)
    prompt_ids = torch.randint(0, 10, (2, 10))
    out = prompt(x, prompt_ids)
    assert out.shape == (2, 10, 128)
    print("✓ LoRA, QLoRA, Adapter, PrefixTuning & PromptTuning")


if __name__ == "__main__":
    test_peft()
