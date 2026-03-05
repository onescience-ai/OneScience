import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from onescience.modules.equivariant.group_conv import GroupEquivariantConv2d, GroupEquivariantConv3d 

class GroupEquivariantMLP2d(nn.Module):
    """
    群等变 2D 多层感知机 (Group Equivariant MLP 2D).

    该模块在群等变特征上执行逐点（Point-wise）的非线性变换。
    它实际上是由两个核大小为 1 的群卷积 (`GConv2d`) 组成的，中间夹着 GELU 激活函数。
    它的作用类似于标准 CNN 中的 1x1 卷积（或 `nn.Linear` 作用于通道），用于在保持几何等变性的前提下混合特征通道。

    Args:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        mid_channels (int): 中间隐藏层的通道数。
        reflection (bool, optional): 是否包含反射群（与输入的 GConv2d配置需一致）。默认值: False。
        last_layer (bool, optional): 最后一层卷积是否执行投影（降维去群维度）。默认值: False。

    形状:
        输入 x: (B, C_in * Group_Size, H, W)。
        输出: (B, C_out * Group_Size, H, W)。如果 last_layer=True，输出为 (B, C_out, H, W)。

    Example:
        >>> # 假设 Group Size = 4 (C4 group)
        >>> gmlp = GroupEquivariantMLP2d(in_channels=32, out_channels=32, mid_channels=64, reflection=False)
        >>> # 输入特征必须包含群维度 (32 * 4 = 128)
        >>> x = torch.randn(2, 128, 64, 64)
        >>> out = gmlp(x)
        >>> print(out.shape)
        torch.Size([2, 128, 64, 64])
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int,
        reflection: bool = False,
        last_layer: bool = False,
    ):
        super(GroupEquivariantMLP2d, self).__init__()
        self.mlp1 = GroupEquivariantConv2d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            reflection=reflection,
        )
        self.mlp2 = GroupEquivariantConv2d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            reflection=reflection,
            last_layer=last_layer,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x


class GroupEquivariantMLP3d(nn.Module):
    """
    群等变 3D 多层感知机 (Group Equivariant MLP 3D).

    `GMLP2d` 的 3D 扩展版本，用于处理 3D 体素数据或点云网格特征。
    它使用 `GConv3d`（核大小为 1）来实现通道混合。适用于 3D 流体模拟、气象预测等任务中的特征提取。

    Args:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        mid_channels (int): 中间隐藏层的通道数。
        reflection (bool, optional): 是否包含反射群。默认值: False。
        last_layer (bool, optional): 是否为最后一层。默认值: False。

    形状:
        输入 x: (B, C_in * Group_Size, D, H, W)。
        输出: (B, C_out * Group_Size, D, H, W)。如果 last_layer=True，输出为 (B, C_out, D, H, W)。

    Example:
        >>> # 假设存在 GConv3d 且 Group Size = 4
        >>> gmlp3d = GroupEquivariantMLP3d(in_channels=16, out_channels=16, mid_channels=32)
        >>> # 输入特征 (Batch, Channels*Group, D, H, W)
        >>> x = torch.randn(2, 16*4, 32, 32, 32)
        >>> out = gmlp3d(x)
        >>> print(out.shape)
        torch.Size([2, 64, 32, 32, 32])
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int,
        reflection: bool = False,
        last_layer: bool = False,
    ):
        super(GroupEquivariantMLP3d, self).__init__()
        self.mlp1 = GroupEquivariantConv3d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            reflection=reflection,
        )
        self.mlp2 = GroupEquivariantConv3d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            reflection=reflection,
            last_layer=last_layer,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x