import torch
import torch.nn as nn
from timm.layers import trunc_normal_
from .Embedding import timestep_embedding, unified_pos_embedding
import numpy as np
import torch.nn.functional as F


################################################################
# Multiscale modules 1D
################################################################


class DoubleConv1D(nn.Module):
    """
    一维双卷积块。

    该模块包含两个连续的卷积层，每个卷积层后通常接归一化层 (BatchNorm/InstanceNorm) 和 ReLU 激活函数。
    结构为：(Conv1d => [BN/IN] => ReLU) * 2。这种结构常用于 U-Net 等架构中，用于增加网络的深度和非线性特征提取能力。

    Args:
        in_channels (int): 输入张量的通道数。
        out_channels (int): 输出张量的通道数。
        mid_channels (int, optional): 中间层的通道数。如果为 None，则默认等于 out_channels。
        normtype (str, optional): 归一化类型。支持 'bn' (BatchNorm1d), 'in' (InstanceNorm1d)。如果为其他值，则不使用归一化。默认为 'bn'。

    形状:
        输入: (B, C_in, L)
        输出: (B, C_out, L)

    Example:
        >>> # 定义一个输入通道为64，输出通道为32的双卷积层
        >>> dconv = DoubleConv1D(64, 32)
        >>> x = torch.randn(8, 64, 100)
        >>> out = dconv(x)
        >>> out.shape
        torch.Size([8, 32, 100])
    """

    def __init__(self, in_channels, out_channels, mid_channels=None, normtype="bn"):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if normtype == "bn":
            self.double_conv = nn.Sequential(
                nn.Conv1d(
                    in_channels, mid_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.BatchNorm1d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(
                    mid_channels, out_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
            )
        elif normtype == "in":
            self.double_conv = nn.Sequential(
                nn.Conv1d(
                    in_channels, mid_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.InstanceNorm1d(mid_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv1d(
                    mid_channels, out_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.InstanceNorm1d(out_channels, affine=True),
                nn.ReLU(inplace=True),
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv1d(
                    in_channels, mid_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.ReLU(inplace=True),
                nn.Conv1d(
                    mid_channels, out_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.double_conv(x)


class Down1D(nn.Module):
    """
        一维下采样模块。

        首先通过最大池化层（MaxPool1d）将空间维度减半，然后应用 DoubleConv1D 提取特征。
        这是编码器（Encoder）路径中的标准组件。

        Args:
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数。
            normtype (str, optional): 归一化类型 ('bn' 或 'in')。默认为 'bn'。

        形状:
            输入: (B, C_in, L)
            输出: (B, C_out, L / 2)

        Example:
            >>> down = Down1D(64, 128)
            >>> x = torch.randn(8, 64, 100)
            >>> out = down(x)
            >>> out.shape
            torch.Size([8, 128, 50])
    """

    def __init__(self, in_channels, out_channels, normtype="bn"):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2), DoubleConv1D(in_channels, out_channels, normtype=normtype)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up1D(nn.Module):
    """
        一维上采样模块。

        该模块首先对输入 x1 进行上采样（线性插值或转置卷积），然后与来自编码器路径的跳跃连接（Skip Connection）特征 x2 进行拼接（Concatenate），
        最后通过 DoubleConv1D 融合特征。这是解码器（Decoder）路径中的关键组件。

        Args:
            in_channels (int): 拼接后的总输入通道数（即 x1 上采样后的通道 + x2 的通道）。通常在 U-Net 设计中，x1 和 x2 通道数各占一半。
            out_channels (int): 输出通道数。
            bilinear (bool, optional): 上采样方式。如果为 True，使用线性插值 (scale_factor=2)；如果为 False，使用转置卷积 (ConvTranspose1d)。默认为 True。
            normtype (str, optional): 归一化类型。默认为 'bn'。

        形状:
            输入 x1 (来自深层): (B, C_1, L)
            输入 x2 (跳跃连接): (B, C_2, 2L)
            输出: (B, C_out, 2L)
            注意: in_channels 参数应等于 C_1 + C_2 (在 bilinear=True 时) 或转置卷积后的通道和。

        Example:
            >>> up = Up1D(128, 64, bilinear=True)
            >>> x1 = torch.randn(8, 64, 50)  # 来自下层的特征
            >>> x2 = torch.randn(8, 64, 100) # 跳跃连接的特征
            >>> out = up(x1, x2)
            >>> out.shape
            torch.Size([8, 64, 100])
    """

    def __init__(self, in_channels, out_channels, bilinear=True, normtype="bn"):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
            self.conv = DoubleConv1D(
                in_channels, out_channels, in_channels // 2, normtype=normtype
            )
        else:
            self.up = nn.ConvTranspose1d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv1D(in_channels, out_channels, normtype=normtype)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv1D(nn.Module):
    """
        一维输出卷积层。

        使用 1x1 卷积将特征图映射到最终的输出通道（例如分类数或回归目标数）。

        Args:
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数。

        形状:
            输入: (B, C_in, L)
            输出: (B, C_out, L)

        Example:
            >>> out_conv = OutConv1D(64, 10) # 假设有10个类别
            >>> x = torch.randn(8, 64, 100)
            >>> out = out_conv(x)
            >>> out.shape
            torch.Size([8, 10, 100])
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv1D, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


################################################################
# Multiscale modules 2D
################################################################
class DoubleConv2D(nn.Module):
    """
        二维双卷积块。

        结构为：(Conv2d => [BN/IN] => ReLU) * 2。
        使用 3x3 卷积核和 padding=1 以保持特征图的空间尺寸不变，专注于特征提取。

        Args:
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数。
            mid_channels (int, optional): 中间层通道数。默认为 None (即等于 out_channels)。
            normtype (str, optional): 归一化类型 ('bn', 'in')。默认为 'bn'。

        形状:
            输入: (B, C_in, H, W)
            输出: (B, C_out, H, W)

        Example:
            >>> dconv = DoubleConv2D(64, 32)
            >>> x = torch.randn(8, 64, 128, 128)
            >>> out = dconv(x)
            >>> out.shape
            torch.Size([8, 32, 128, 128])
    """

    def __init__(self, in_channels, out_channels, mid_channels=None, normtype="bn"):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if normtype == "bn":
            self.double_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels, mid_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    mid_channels, out_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        elif normtype == "in":
            self.double_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels, mid_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.InstanceNorm2d(mid_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    mid_channels, out_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.InstanceNorm2d(out_channels, affine=True),
                nn.ReLU(inplace=True),
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels, mid_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    mid_channels, out_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.double_conv(x)


class Down2D(nn.Module):
    """
        二维下采样模块。

        使用 2x2 最大池化（MaxPool2d）将高宽减半，随后连接 DoubleConv2D 进行特征提取。

        Args:
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数。
            normtype (str, optional): 归一化类型。

        形状:
            输入: (B, C_in, H, W)
            输出: (B, C_out, H/2, W/2)

        Example:
            >>> down = Down2D(64, 128)
            >>> x = torch.randn(8, 64, 128, 128)
            >>> out = down(x)
            >>> out.shape
            torch.Size([8, 128, 64, 64])
    """

    def __init__(self, in_channels, out_channels, normtype="bn"):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv2D(in_channels, out_channels, normtype=normtype)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up2D(nn.Module):
    """
        二维上采样模块。

        对输入 x1 进行上采样（双线性插值或转置卷积），对齐尺寸后与跳跃连接 x2 拼接，再经过卷积融合。
        该模块包含自动 Padding 逻辑，以健壮地处理 x1 和 x2 尺寸因奇偶性导致不完全匹配的情况。

        Args:
            in_channels (int): 拼接后的总输入通道数。
            out_channels (int): 输出通道数。
            bilinear (bool, optional): 是否使用双线性插值上采样。默认为 True。
            normtype (str, optional): 归一化类型。

        形状:
            输入 x1: (B, C_1, H, W)
            输入 x2: (B, C_2, H_target, W_target)，通常 H_target approx 2H
            输出: (B, C_out, H_target, W_target)

        Example:
            >>> up = Up2D(128, 64, bilinear=True) # 假设拼接后通道为128
            >>> x1 = torch.randn(8, 64, 64, 64)
            >>> x2 = torch.randn(8, 64, 128, 128)
            >>> out = up(x1, x2)
            >>> out.shape
            torch.Size([8, 64, 128, 128])
    """

    def __init__(self, in_channels, out_channels, bilinear=True, normtype="bn"):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv2D(
                in_channels, out_channels, in_channels // 2, normtype=normtype
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv2D(in_channels, out_channels, normtype=normtype)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv2D(nn.Module):
    """
        二维输出卷积层。

        使用 1x1 卷积进行通道映射，通常用于将特征维度映射到像素级的分类或回归值。

        Args:
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数。

        形状:
            输入: (B, C_in, H, W)
            输出: (B, C_out, H, W)

        Example:
            >>> out_conv = OutConv2D(64, 3) # 输出RGB图像
            >>> x = torch.randn(8, 64, 128, 128)
            >>> out = out_conv(x)
            >>> out.shape
            torch.Size([8, 3, 128, 128])
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


################################################################
# Multiscale modules 3D
################################################################


class DoubleConv3D(nn.Module):
    """
        三维双卷积块。

        结构为：(Conv3d => [BN/IN] => ReLU) * 2。
        用于处理体素数据（Volumetric Data）或视频数据，在三个空间维度上提取特征。

        Args:
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数。
            mid_channels (int, optional): 中间通道数。
            normtype (str, optional): 归一化类型 ('bn' 为 BatchNorm3d, 'in' 为 InstanceNorm3d)。

        形状:
            输入: (B, C_in, D, H, W)
            输出: (B, C_out, D, H, W)

        Example:
            >>> dconv = DoubleConv3D(64, 32)
            >>> x = torch.randn(4, 64, 32, 64, 64)
            >>> out = dconv(x)
            >>> out.shape
            torch.Size([4, 32, 32, 64, 64])
    """

    def __init__(self, in_channels, out_channels, mid_channels=None, normtype="bn"):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if normtype == "bn":
            self.double_conv = nn.Sequential(
                nn.Conv3d(
                    in_channels, mid_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.BatchNorm3d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(
                    mid_channels, out_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
            )
        elif normtype == "in":
            self.double_conv = nn.Sequential(
                nn.Conv3d(
                    in_channels, mid_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.InstanceNorm3d(mid_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(
                    mid_channels, out_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.InstanceNorm3d(out_channels, affine=True),
                nn.ReLU(inplace=True),
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv3d(
                    in_channels, mid_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.ReLU(inplace=True),
                nn.Conv3d(
                    mid_channels, out_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.double_conv(x)


class Down3D(nn.Module):
    """
        三维下采样模块。

        使用 2x2x2 最大池化（MaxPool3d）将所有空间维度（D, H, W）减半，然后进行双卷积。

        Args:
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数。
            normtype (str, optional): 归一化类型。

        形状:
            输入: (B, C_in, D, H, W)
            输出: (B, C_out, D/2, H/2, W/2)

        Example:
            >>> down = Down3D(32, 64)
            >>> x = torch.randn(4, 32, 32, 64, 64)
            >>> out = down(x)
            >>> out.shape
            torch.Size([4, 64, 16, 32, 32])
    """

    def __init__(self, in_channels, out_channels, normtype="bn"):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2), DoubleConv3D(in_channels, out_channels, normtype=normtype)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    """
        三维上采样模块。

        对输入 x1 进行上采样（三线性插值 Trilinear 或转置卷积），并与跳跃连接 x2 拼接。
        用于在 3D U-Net 的解码路径中逐步恢复体素的空间分辨率。

        Args:
            in_channels (int): 拼接后的总输入通道数。
            out_channels (int): 输出通道数。
            bilinear (bool, optional): 是否使用三线性插值 (mode='trilinear')。默认为 True。
            normtype (str, optional): 归一化类型。

        形状:
            输入 x1: (B, C_1, D, H, W)
            输入 x2: (B, C_2, 2D, 2H, 2W)
            输出: (B, C_out, 2D, 2H, 2W)

        Example:
            >>> up = Up3D(128, 64, bilinear=True)
            >>> x1 = torch.randn(4, 64, 16, 32, 32)
            >>> x2 = torch.randn(4, 64, 32, 64, 64)
            >>> out = up(x1, x2)
            >>> out.shape
            torch.Size([4, 64, 32, 64, 64])
    """

    def __init__(self, in_channels, out_channels, bilinear=True, normtype="bn"):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
            self.conv = DoubleConv3D(
                in_channels, out_channels, in_channels // 2, normtype=normtype
            )
        else:
            self.up = nn.ConvTranspose3d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv3D(in_channels, out_channels, normtype=normtype)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv3D(nn.Module):
    """
        三维输出卷积层。

        使用 1x1x1 卷积将深层特征映射到最终的输出空间（例如分割掩码或预测场）。

        Args:
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数。

        形状:
            输入: (B, C_in, D, H, W)
            输出: (B, C_out, D, H, W)

        Example:
            >>> out_conv = OutConv3D(64, 1) # 例如二分类分割任务
            >>> x = torch.randn(4, 64, 32, 64, 64)
            >>> out = out_conv(x)
            >>> out.shape
            torch.Size([4, 1, 32, 64, 64])
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
