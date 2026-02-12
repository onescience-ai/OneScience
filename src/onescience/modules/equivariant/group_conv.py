import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Union, Tuple

class GConv2d(nn.Module):
    """
    群等变 2D 卷积层 (Group Equivariant Conv2d).

    该模块实现了基于离散群（C4 旋转群或 D4 二面体群）的卷积操作。
    它通过在群元素上共享权重来实现旋转和平移的等变性。

    Args:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        kernel_size (int): 卷积核大小 (必须为奇数)。
        bias (bool, optional): 是否添加偏置。默认值: True。
        first_layer (bool, optional): 是否为第一层（Lifting Layer）。默认值: False。
        last_layer (bool, optional): 是否为最后一层（Projection Layer）。默认值: False。
        spectral (bool, optional): 权重是否为复数。默认值: False。
        Hermitian (bool, optional): 是否强制 Hermitian 对称。默认值: False。
        reflection (bool, optional): 是否包含反射群 (D4)。默认值: False。

    形状:
        输入: (B, C_in * Group_Size, H, W)。(first_layer=True 时除外)
        输出: (B, C_out * Group_Size, H, W)。(last_layer=True 时除外)

    Example:
        >>> # 1. 第一层 (Lifting)
        >>> gconv = GConv2d(32, 64, kernel_size=3, first_layer=True)
        >>> x = torch.randn(2, 32, 64, 64)
        >>> out = gconv(x)
        >>> print(out.shape) # 32 -> 64*4
        torch.Size([2, 256, 64, 64])
        >>>
        >>> # 2. 中间层 (Group Conv)
        >>> gconv2 = GConv2d(64, 64, kernel_size=3)
        >>> out2 = gconv2(out)
        >>> print(out2.shape)
        torch.Size([2, 256, 64, 64])
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        bias: bool = True,
        first_layer: bool = False,
        last_layer: bool = False,
        spectral: bool = False,
        Hermitian: bool = False,
        reflection: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reflection = reflection
        self.rt_group_size = 4
        self.group_size = self.rt_group_size * (1 + reflection)
        
        assert kernel_size % 2 == 1, f"kernel size must be odd, got {kernel_size}"
        
        dtype = torch.cfloat if spectral else torch.float
        self.kernel_size_Y = kernel_size
        self.kernel_size_X = kernel_size // 2 + 1 if Hermitian else kernel_size
        self.Hermitian = Hermitian
        self.first_layer = first_layer
        self.last_layer = last_layer
        self.spectral = spectral

        if first_layer or last_layer:
            self.W = nn.Parameter(
                torch.empty(out_channels, 1, in_channels, self.kernel_size_Y, self.kernel_size_X, dtype=dtype)
            )
        else:
            if self.Hermitian:
                self.W = nn.ParameterDict({
                    "y0_modes": nn.Parameter(torch.empty(out_channels, 1, in_channels, self.group_size, self.kernel_size_X - 1, 1, dtype=dtype)),
                    "yposx_modes": nn.Parameter(torch.empty(out_channels, 1, in_channels, self.group_size, self.kernel_size_Y, self.kernel_size_X - 1, dtype=dtype)),
                    "00_modes": nn.Parameter(torch.empty(out_channels, 1, in_channels, self.group_size, 1, 1, dtype=torch.float)),
                })
            else:
                self.W = nn.Parameter(torch.empty(out_channels, 1, in_channels, self.group_size, self.kernel_size_Y, self.kernel_size_X, dtype=dtype))

        self.B = nn.Parameter(torch.empty(1, out_channels, 1, 1)) if bias else None
        self.reset_parameters()
        self.get_weight()

    def reset_parameters(self):
        if isinstance(self.W, nn.ParameterDict):
            for v in self.W.values():
                nn.init.kaiming_uniform_(v, a=math.sqrt(5))
        else:
            nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        if self.B is not None:
            nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))

    def get_weight(self):
        if self.Hermitian:
            weights = torch.cat([
                self.W["y0_modes"],
                self.W["00_modes"].cfloat(),
                self.W["y0_modes"].flip(dims=(-2,)).conj(),
            ], dim=-2)
            weights = torch.cat([weights, self.W["yposx_modes"]], dim=-1)
            weights = torch.cat([weights[..., 1:].conj().rot90(k=2, dims=[-2, -1]), weights], dim=-1)
        else:
            weights = self.W

        if self.first_layer or self.last_layer:
            weights = weights.repeat(1, self.group_size, 1, 1, 1)
            for k in range(1, self.rt_group_size):
                weights[:, k] = weights[:, k].rot90(k=k, dims=[-2, -1])
            if self.reflection:
                weights[:, self.rt_group_size:] = weights[:, :self.rt_group_size].flip(dims=[-2])

            if self.first_layer:
                weights = weights.view(self.out_channels * self.group_size, self.in_channels, self.kernel_size_Y, self.kernel_size_Y)
                if self.B is not None:
                    self.bias = self.B.repeat_interleave(repeats=self.group_size, dim=1).view(-1)
                else:
                    self.bias = None
            else:
                weights = weights.transpose(2, 1).reshape(self.out_channels, self.in_channels * self.group_size, self.kernel_size_Y, self.kernel_size_Y)
                if self.B is not None:
                    self.bias = self.B.view(-1)
                else:
                    self.bias = None
        else:
            weights = weights.repeat(1, self.group_size, 1, 1, 1, 1)
            for k in range(1, self.rt_group_size):
                weights[:, k] = weights[:, k - 1].rot90(dims=[-2, -1])
                # Dim 2 is Group Dimension. Slicing logic for Permutation:
                if self.reflection:
                    weights[:, k] = torch.cat([
                        weights[:, k, :, self.rt_group_size - 1].unsqueeze(2),
                        weights[:, k, :, : (self.rt_group_size - 1)],
                        weights[:, k, :, (self.rt_group_size + 1) :],
                        weights[:, k, :, self.rt_group_size].unsqueeze(2),
                    ], dim=2)
                else:
                    weights[:, k] = torch.cat([
                        weights[:, k, :, -1].unsqueeze(2),
                        weights[:, k, :, :-1],
                    ], dim=2)

            if self.reflection:
                weights[:, self.rt_group_size:] = torch.cat([
                    weights[:, :self.rt_group_size, :, self.rt_group_size:],
                    weights[:, :self.rt_group_size, :, :self.rt_group_size],
                ], dim=3).flip([-2])

            weights = weights.view(self.out_channels * self.group_size, self.in_channels * self.group_size, self.kernel_size_Y, self.kernel_size_Y)
            if self.B is not None:
                self.bias = self.B.repeat_interleave(repeats=self.group_size, dim=1).view(-1)
            else:
                self.bias = None

        if self.Hermitian:
            weights = weights[..., -self.kernel_size_X:]
            
        self.weights = weights
        return self.weights

    def forward(self, x):
        self.get_weight()
        return F.conv2d(input=x, weight=self.weights, bias=self.bias)


class GConv3d(nn.Module):
    """
    群等变 3D 卷积层 (Group Equivariant Conv3d).

    

    实现了具有圆柱对称性 (Cylindrical Symmetry) 的 3D 群卷积。
    假设群操作（旋转/反射）仅作用于空间维度 H (Y) 和 W (X)，而深度维度 D (Z) 保持平移等变性。

    Args:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        kernel_size (int or tuple): 卷积核大小 (Kd, Kh, Kw)。必须为奇数。
        bias (bool, optional): 是否添加偏置。
        first_layer (bool, optional): 是否为 Lifting Layer。
        last_layer (bool, optional): 是否为 Projection Layer。
        reflection (bool, optional): 是否包含反射群 (D4)。

    形状:
        输入: (B, C_in * Group_Size, D, H, W)。(first_layer=True 时除外)
        输出: (B, C_out * Group_Size, D, H, W)。(last_layer=True 时除外)

    Example:
        >>> # 1. 3D Lifting Layer (scalar -> group)
        >>> gconv3d = GConv3d(16, 32, kernel_size=(3, 3, 3), first_layer=True)
        >>> x = torch.randn(2, 16, 10, 32, 32) # (B, C, D, H, W)
        >>> out = gconv3d(x)
        >>> print(out.shape) # 32 * 4 = 128
        torch.Size([2, 128, 10, 32, 32])
        >>>
        >>> # 2. 3D Group Conv (group -> group)
        >>> gconv3d_mid = GConv3d(32, 32, kernel_size=3)
        >>> out2 = gconv3d_mid(out)
        >>> print(out2.shape) # 32*4 -> 32*4
        torch.Size([2, 128, 10, 32, 32])
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        bias: bool = True,
        first_layer: bool = False,
        last_layer: bool = False,
        reflection: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reflection = reflection
        self.rt_group_size = 4
        self.group_size = self.rt_group_size * (1 + reflection)

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        
        assert all(k % 2 == 1 for k in self.kernel_size), "3D Kernel sizes must be odd"

        self.first_layer = first_layer
        self.last_layer = last_layer

        # 3D 权重构造：增加一个维度
        if first_layer or last_layer:
             self.W = nn.Parameter(torch.empty(
                 out_channels, 1, in_channels, 
                 self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]
             ))
        else:
             # 中间层: (Out, 1, In, Group, Kd, Kh, Kw)
             self.W = nn.Parameter(
                torch.empty(out_channels, 1, in_channels, self.group_size, *self.kernel_size)
             )

        self.B = nn.Parameter(torch.empty(1, out_channels, 1, 1, 1)) if bias else None
        
        self.reset_parameters()
        self.get_weight()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        if self.B is not None:
            nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))

    def get_weight(self):
        weights = self.W

        if self.first_layer or self.last_layer:
            weights = weights.repeat(1, self.group_size, 1, 1, 1, 1)
            # 旋转仅作用于 H(-2) 和 W(-1)
            for k in range(1, self.rt_group_size):
                weights[:, k] = weights[:, k].rot90(k=k, dims=[-2, -1])

            if self.reflection:
                weights[:, self.rt_group_size:] = weights[:, :self.rt_group_size].flip(dims=[-2])

            if self.first_layer:
                weights = weights.view(self.out_channels * self.group_size, self.in_channels, *self.kernel_size)
                if self.B is not None:
                    self.bias = self.B.repeat_interleave(repeats=self.group_size, dim=1).view(-1)
                else:
                    self.bias = None
            else:
                weights = weights.transpose(2, 1).reshape(self.out_channels, self.in_channels * self.group_size, *self.kernel_size)
                if self.B is not None:
                    self.bias = self.B.view(-1)
                else:
                    self.bias = None

        else:
            # Middle Layer: (Out, 1, In, Group, Kd, Kh, Kw)
            weights = weights.repeat(1, self.group_size, 1, 1, 1, 1, 1)

            for k in range(1, self.rt_group_size):
                weights[:, k] = weights[:, k - 1].rot90(dims=[-2, -1]) # Rotate H, W

                # 【核心修复】：群维度在索引 2，必须在 dim=2 上进行拼接，而不是 dim=3 (Depth)
                if self.reflection:
                    weights[:, k] = torch.cat([
                        weights[:, k, :, self.rt_group_size - 1].unsqueeze(2),
                        weights[:, k, :, : (self.rt_group_size - 1)],
                        weights[:, k, :, (self.rt_group_size + 1) :],
                        weights[:, k, :, self.rt_group_size].unsqueeze(2),
                    ], dim=2)
                else:
                    weights[:, k] = torch.cat([
                        weights[:, k, :, -1].unsqueeze(2),
                        weights[:, k, :, :-1],
                    ], dim=2)

            if self.reflection:
                # 反射操作：翻转输出群，同时翻转输入群(dim 4 is G_in) 和 空间核
                # 注意：3D中 G_in 是 dim 4，Kh 是 dim -2
                weights[:, self.rt_group_size:] = torch.cat([
                    weights[:, :self.rt_group_size, :, self.rt_group_size:],
                    weights[:, :self.rt_group_size, :, :self.rt_group_size],
                ], dim=4).flip([-2])

            weights = weights.view(self.out_channels * self.group_size, self.in_channels * self.group_size, *self.kernel_size)
            
            if self.B is not None:
                self.bias = self.B.repeat_interleave(repeats=self.group_size, dim=1).view(-1)
            else:
                self.bias = None

        self.weights = weights
        return self.weights

    def forward(self, x):
        self.get_weight()
        return F.conv3d(input=x, weight=self.weights, bias=self.bias)