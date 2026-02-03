import torch
import torch.nn as nn


from onescience.registry import Geometry

@Geometry.registry_module()
class BackboneBuilder(nn.Module):
    """
    BackboneBuilder
    ======================================================
    功能:
        - 根据残基特征和刚体变换构建蛋白质主链骨架坐标；
        - 通常输入来自 PairUpdate / TriangleAttention / RigidTransform；
        - 输出主链原子坐标 (N, CA, C)。
    
    输入:
        rigid_transforms: List[RigidTransform] 每个残基的刚体变换
        atom_mask: [B, L, 3] 每个原子是否存在
    输出:
        backbone_coords: [B, L, 3, 3] 3个原子坐标 (N, CA, C)
    """

    def __init__(self):
        super().__init__()

    def forward(self, rigid_transforms, atom_mask=None):
        """
        rigid_transforms: List[RigidTransform] 长度=L
        atom_mask: [B, L, 3] 可选
        """
        B = rigid_transforms[0].R.shape[0]
        L = len(rigid_transforms)
        device = rigid_transforms[0].R.device

        # 初始化坐标张量 [B, L, 3, 3]
        backbone_coords = torch.zeros(B, L, 3, 3, device=device)

        for i, rigid in enumerate(rigid_transforms):
            # 局部标准坐标 (N, CA, C)
            local_coords = torch.tensor([
                [1.458, 0.0, 0.0],   # N
                [0.0, 1.525, 0.0],   # CA
                [0.0, 0.0, 1.525]    # C
            ], device=device).unsqueeze(0).repeat(B,1,1)  # [B,3,3]

            # 应用刚体变换
            transformed = rigid.apply(local_coords)  # [B,3,3]
            backbone_coords[:, i, :, :] = transformed

        # 可选 mask
        if atom_mask is not None:
            backbone_coords = backbone_coords * atom_mask.unsqueeze(-1)

        return backbone_coords


if __name__ == "__main__":
    B, L = 2, 128
    device = "cpu"

    # 构造示例刚体变换
    R = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(B, L, 1, 1)
    t = torch.zeros(B, L, 3)
    rigid_transforms = [RigidTransform(R[:, i], t[:, i]) for i in range(L)]

    builder = BackboneBuilder()
    backbone_coords = builder(rigid_transforms)
    print("Backbone coords shape:", backbone_coords.shape)  # [2,128,3,3]
