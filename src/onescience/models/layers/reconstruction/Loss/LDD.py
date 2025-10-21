import torch
import torch.nn as nn

from onescience.registry import Loss

@Loss.registry_module()
class LDDTLoss(nn.Module):
    """
    Local Distance Difference Test Loss (lDDT)
    ======================================================
    功能:
        - 衡量预测结构与真实结构的局部原子距离一致性
        - 屏蔽缺失原子/残基
        - 用于蛋白质结构预测辅助监督
    
    输入:
        pred_coords: [B, L, N, 3] 预测原子坐标
        true_coords: [B, L, N, 3] 真实原子坐标
        mask: [B, L, N] 原子存在掩码
        cutoff: float 局部距离阈值 (Å)
    输出:
        lddt_loss: 标量
    """

    def __init__(self, cutoff=15.0, eps=1e-6):
        super().__init__()
        self.cutoff = cutoff
        self.eps = eps

    def forward(self, pred_coords, true_coords, mask=None):
        """
        pred_coords, true_coords: [B,L,N,3]
        mask: [B,L,N] 可选
        """
        B, L, N, _ = pred_coords.shape

        # 计算预测原子间距离矩阵 [B,L,N,N]
        pred_dist = torch.cdist(pred_coords.view(B*L, N, 3), pred_coords.view(B*L, N, 3))  # [B*L,N,N]
        true_dist = torch.cdist(true_coords.view(B*L, N, 3), true_coords.view(B*L, N, 3))

        # 局部距离差
        diff = (pred_dist - true_dist).abs()  # [B*L,N,N]

        # lDDT评分：根据阈值打分
        score = (diff < 0.5).float() * 1.0 + \
                ((diff >= 0.5) & (diff < 1.0)).float() * 0.8 + \
                ((diff >= 1.0) & (diff < 2.0)).float() * 0.6 + \
                ((diff >= 2.0) & (diff < 4.0)).float() * 0.4

        # mask 原子
        if mask is not None:
            mask_2d = mask.unsqueeze(-1) * mask.unsqueeze(-2)  # [B,L,N,N]
            score = score.view(B, L, N, N) * mask_2d

        lddt_loss = 1.0 - score.mean()
        return lddt_loss


if __name__ == "__main__":
    B, L, N = 2, 5, 3
    pred = torch.randn(B, L, N, 3)
    true = torch.randn(B, L, N, 3)
    mask = torch.ones(B, L, N)

    lddt_loss_fn = LDDTLoss()
    loss = lddt_loss_fn(pred, true, mask)
    print("lDDT loss:", loss.item())
