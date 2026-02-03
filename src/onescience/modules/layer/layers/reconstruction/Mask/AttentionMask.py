import torch
import torch.nn as nn

from onescience.registry import Mask

@Mask.registry_module()
class AttentionMask:
    """
    AttentionMask
    ======================================================
    功能:
        - 根据残基掩码 / pair 掩码生成注意力 mask
        - 屏蔽 padding 或无效残基对
        - 输出可直接用于 torch.nn.MultiheadAttention 或自定义 attention
    """

    @staticmethod
    def build_attention_mask(residue_mask):
        """
        residue_mask: [B, L] 1=有效残基, 0=padding
        return: [B, 1, L, L] 注意力 mask, 1=可 attend, 0=屏蔽
        """
        pair_mask = residue_mask.unsqueeze(-1) * residue_mask.unsqueeze(-2)  # [B,L,L]
        attn_mask = pair_mask.unsqueeze(1)  # [B,1,L,L] 可广播到 heads
        return attn_mask

    @staticmethod
    def apply_mask(attn_logits, attn_mask):
        """
        将注意力 logits 与 mask 结合
        attn_logits: [B, H, L, L]
        attn_mask: [B,1,L,L]
        """
        # mask=0的位置填充负无穷，使 softmax 后权重为0
        masked_logits = attn_logits.masked_fill(attn_mask == 0, float("-inf"))
        return masked_logits

if __name__ == "__main__":
    B, L, H = 2, 5, 8
    residue_mask = torch.tensor([
        [1,1,1,0,0],
        [1,1,0,0,0]
    ])

    # 构建注意力 mask
    attn_mask = AttentionMask.build_attention_mask(residue_mask)
    print("Attention mask shape:", attn_mask.shape)  # [B,1,L,L]

    # 模拟注意力 logits
    attn_logits = torch.randn(B, H, L, L)
    masked_logits = AttentionMask.apply_mask(attn_logits, attn_mask)
    print("Masked logits shape:", masked_logits.shape)
