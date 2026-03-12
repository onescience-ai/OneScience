"""VLM 多模态模块"""

import torch
import torch.nn as nn
from .transformer import TransformerBlock
from .rag import CrossAttention


class VisionEncoder(nn.Module):
    """视觉编码器"""
    def __init__(self, patch_size=16, dim=512, depth=6, heads=8):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads, dim * 4) for _ in range(depth)])

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        return x


class MultiModalFusion(nn.Module):
    """多模态融合"""
    def __init__(self, dim, heads):
        super().__init__()
        self.cross_attn = CrossAttention(dim, heads)
        self.gate = nn.Linear(dim * 2, dim)

    def forward(self, text, vision):
        fused = self.cross_attn(text, vision)
        gate = torch.sigmoid(self.gate(torch.cat([text, fused], dim=-1)))
        return text + gate * fused


def test_vlm():
    print("\n[测试] VLM")

    vision = VisionEncoder(patch_size=16, dim=128, depth=2, heads=4)
    img = torch.randn(2, 3, 224, 224)
    out = vision(img)
    assert out.shape == (2, 196, 128)

    fusion = MultiModalFusion(128, 4)
    text = torch.randn(2, 10, 128)
    vision_emb = torch.randn(2, 196, 128)
    out = fusion(text, vision_emb)
    assert out.shape == (2, 10, 128)
    print("✓ VisionEncoder & MultiModalFusion")


if __name__ == "__main__":
    test_vlm()
