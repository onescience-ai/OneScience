import torch
import torch.nn as nn

from onescience.registry import Loss

@Loss.registry_module()
class FAPE(nn.Module):
    """
    Frame Aligned Point Error (FAPE)
    ======================================================
    功能:
        - 计算预测原子坐标与真实原子坐标在残基局部坐标系下的误差
        - 用于蛋白质结构模块训练
        - 屏蔽缺失原子/残基

    输入:
        pred_coords: [B, L, N, 3] 预测原子坐标 (N=原子数, e.g., N, CA, C)
        true_coords: [B, L, N, 3] 真实原子坐标
        rigid_transforms: List[RigidTransform] 残基局部坐标系
        mask: [B, L, N] 原子存在掩码
    输出:
        fape_loss: 标量
    """

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred_coords, true_coords, rigid_transforms, mask=None):
        """
        pred_coords, true_coords: [B,L,N,3]
        rigid_transforms: List[RigidTransform] 长度=L
        mask: [B,L,N]
        """
        B, L, N, _ = pred_coords.shape
        device = pred_coords.device

        # 将坐标映射到每个残基局部坐标系
        local_pred = torch.zeros_like(pred_coords)
        local_true = torch.zeros_like(true_coords)

        for i in range(L):
            rigid = rigid_transforms[i]
            inv = rigid.invert()
            local_pred[:, i] = inv.apply(pred_coords[:, i])
            local_true[:, i] = inv.apply(true_coords[:, i])

        # 计算欧氏距离误差
        diff = local_pred - local_true  # [B,L,N,3]
        dist = torch.sqrt((diff**2).sum(dim=-1) + self.eps)  # [B,L,N]

        # mask 原子
        if mask is not None:
            dist = dist * mask

        fape_loss = dist.sum() / (mask.sum() + self.eps) if mask is not None else dist.mean()
        return fape_loss


if __name__ == "__main__":
    B, L, N = 2, 128, 3
    pred = torch.randn(B, L, N, 3)
    true = torch.randn(B, L, N, 3)

    # 构造刚体变换（单位旋转+零平移）
    R = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(B, L, 1, 1)
    t = torch.zeros(B, L, 3)
    rigid_transforms = [RigidTransform(R[:, i], t[:, i]) for i in range(L)]

    mask = torch.ones(B, L, N)
    fape = FAPE()
    loss = fape(pred, true, rigid_transforms, mask)
    print("FAPE loss:", loss.item())
