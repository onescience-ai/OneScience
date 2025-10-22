import torch
import torch.nn as nn

from onescience.registry import Geometry

@Geometry.registry_module()
class RigidTransform:
    """
    Rigid Transformation (Rotation + Translation)
    ==================================================
    功能：
        - 表示刚体变换，将局部坐标映射到全局坐标；
        - 通常用于残基主链或侧链坐标预测。

    结构：
        R: [B, L, 3, 3] 旋转矩阵
        t: [B, L, 3]     平移向量
    """

    def __init__(self, rotation: torch.Tensor, translation: torch.Tensor):
        """
        rotation: [B, L, 3, 3]
        translation: [B, L, 3]
        """
        assert rotation.shape[-2:] == (3, 3)
        assert translation.shape[-1] == 3
        self.R = rotation
        self.t = translation

    def apply(self, coords: torch.Tensor):
        """
        将局部坐标 coords 应用刚体变换
        coords: [B, L, N, 3] 局部坐标 (B=batch, L=残基, N=原子数)
        返回：
            transformed: [B, L, N, 3]
        """
        # coords @ R^T + t
        transformed = torch.einsum('blij,blnj->blni', self.R, coords) + self.t.unsqueeze(-2)
        return transformed

    def invert(self):
        """
        返回逆变换
        """
        R_inv = self.R.transpose(-2, -1)
        t_inv = -torch.einsum('blij,blj->bli', R_inv, self.t)
        return RigidTransform(R_inv, t_inv)

    def compose(self, other: "RigidTransform"):
        """
        组合两个刚体变换：self ∘ other
        """
        R_new = torch.matmul(self.R, other.R)
        t_new = torch.einsum('blij,blj->bli', self.R, other.t) + self.t
        return RigidTransform(R_new, t_new)


if __name__ == "__main__":
    B, L, N = 2, 128, 3
    R = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(B, L, 1, 1)  # 单位旋转
    t = torch.zeros(B, L, 3)  # 零平移

    rigid = RigidTransform(R, t)

    coords = torch.randn(B, L, N, 3)
    transformed = rigid.apply(coords)
    print("Transformed coords shape:", transformed.shape)  # [2,128,3,3]

    # 逆变换
    inv = rigid.invert()
    restored = inv.apply(transformed)
    print("Restored difference:", (restored - coords).abs().max())
