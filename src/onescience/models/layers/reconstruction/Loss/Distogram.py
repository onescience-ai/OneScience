import torch
import torch.nn as nn
import torch.nn.functional as F

from onescience.registry import Loss

@Loss.registry_module()
class Distogram(nn.Module):
    """
    Distogram prediction module
    ======================================================
    功能:
        - 输入残基对特征，预测残基间距离概率分布
        - 输出可用于重建3D骨架或计算Distogram Loss
    """

    def __init__(self, pair_dim=256, num_bins=64):
        super().__init__()
        self.num_bins = num_bins
        # 输出每对残基的距离分布
        self.distogram_head = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, pair_dim),
            nn.ReLU(),
            nn.Linear(pair_dim, num_bins)
        )

    def forward(self, pair_repr):
        """
        pair_repr: [B, L, L, C]
        return: [B, L, L, num_bins] 残基对距离概率分布
        """
        logits = self.distogram_head(pair_repr)
        dist_probs = F.softmax(logits, dim=-1)
        return dist_probs

if __name__ == "__main__":
    B, L, C, D = 2, 128, 256, 64
    pair_repr = torch.randn(B, L, L, C)

    distogram = Distogram(pair_dim=C, num_bins=D)
    dist_probs = distogram(pair_repr)
    print("Distogram shape:", dist_probs.shape)  # [2,128,128,64]

    # 取期望距离
    bins = torch.linspace(2.0, 20.0, D)  # 假设 bin 范围 2~20 Å
    expected_dist = (dist_probs * bins.view(1, 1, 1, -1)).sum(-1)
    print("Expected distance shape:", expected_dist.shape)  # [2,128,128]
