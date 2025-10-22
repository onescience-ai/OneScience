import torch
import torch.nn as nn

from onescience.registry import Geometry

@Geometry.registry_module()
class Quaternion2RotMat(nn.Module):
    """
    Convert quaternion to rotation matrix
    ======================================================
    输入:
        quat: [B, L, 4] 四元数 (w, x, y, z)
    输出:
        R: [B, L, 3, 3] 旋转矩阵
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, quat):
        """
        quat: [B, L, 4] (w, x, y, z)
        """
        B, L, _ = quat.shape
        # 归一化四元数
        quat = quat / (quat.norm(dim=-1, keepdim=True) + self.eps)
        w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

        # 构建旋转矩阵
        R = torch.zeros(B, L, 3, 3, device=quat.device, dtype=quat.dtype)

        R[..., 0, 0] = 1 - 2 * (y**2 + z**2)
        R[..., 0, 1] = 2 * (x*y - z*w)
        R[..., 0, 2] = 2 * (x*z + y*w)

        R[..., 1, 0] = 2 * (x*y + z*w)
        R[..., 1, 1] = 1 - 2 * (x**2 + z**2)
        R[..., 1, 2] = 2 * (y*z - x*w)

        R[..., 2, 0] = 2 * (x*z - y*w)
        R[..., 2, 1] = 2 * (y*z + x*w)
        R[..., 2, 2] = 1 - 2 * (x**2 + y**2)

        return R


if __name__ == "__main__":
    B, L = 2, 5
    quat = torch.randn(B, L, 4)  # 随机四元数
    quat2rot = Quaternion2RotMat()
    R = quat2rot(quat)
    print("R shape:", R.shape)  # [2,5,3,3]

    # 验证旋转矩阵正交性 R*R^T ≈ I
    I = torch.matmul(R, R.transpose(-2, -1))
    print("Orthogonality error:", (I - torch.eye(3)).abs().max())
