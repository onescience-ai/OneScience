import torch
import torch.nn as nn
from onescience.modules.layer.unet_layer import (
    DoubleConv1D, Down1D, 
    DoubleConv2D, Down2D, 
    DoubleConv3D, Down3D
)

class UNetEncoder1D(nn.Module):
    """
    一维 U-Net 编码器 (UNet Encoder 1D)。

    

    负责提取一维序列信号的多尺度特征。该模块由一个初始双卷积层和多个下采样层（Down1D）组成。
    在每次前向传播时，会返回一个包含所有层级特征的列表，以供解码器进行跳跃连接（Skip Connection）。

    Args:
        in_channels (int): 输入通道数。
        base_channels (int, optional): 初始特征通道数。默认值: 16。
        num_stages (int, optional): 下采样的层数。默认值: 2。
        bilinear (bool, optional): 是否在解码器中使用双线性插值。默认值: True。
        normtype (str, optional): 归一化类型 ('bn' 或 'in')。默认值: 'bn'。
        kernel_size (int, optional): 卷积核大小，必须为奇数。默认值: 3。

    形状:
        输入 x: (B, C_in, L)
        输出: 一个列表，包含 (num_stages + 1) 个 Tensor。最后一个 Tensor 为最深层瓶颈特征。

    Example:
        >>> encoder = UNetEncoder1D(in_channels=3, base_channels=64, num_stages=4, kernel_size=5)
        >>> x = torch.randn(8, 3, 128)
        >>> features = encoder(x)
        >>> print([f.shape for f in features])
        [torch.Size([8, 64, 128]), torch.Size([8, 128, 64]), torch.Size([8, 256, 32]), torch.Size([8, 512, 16]), torch.Size([8, 512, 8])]
    """
    def __init__(self, in_channels, base_channels=16, num_stages=2, bilinear=True, normtype="bn", kernel_size=3):
        super().__init__()
        self.inc = DoubleConv1D(in_channels, base_channels, normtype=normtype, kernel_size=kernel_size)
        self.down_stages = nn.ModuleList()
        in_ch = base_channels
        for i in range(num_stages):
            out_ch = in_ch * 2
            self.down_stages.append(Down1D(in_ch, out_ch, normtype=normtype, kernel_size=kernel_size))
            in_ch = out_ch

    def forward(self, x):
        features = [self.inc(x)]
        for stage in self.down_stages:
            features.append(stage(features[-1]))
        return features


class UNetEncoder2D(nn.Module):
    """
    二维 U-Net 编码器 (UNet Encoder 2D)。

    负责提取二维网格或图像的多尺度特征。由初始 DoubleConv2D 和多个 Down2D 组成。
    返回包含多层级特征的列表。

    Args:
        in_channels (int): 输入通道数。
        base_channels (int, optional): 初始特征通道数。默认值: 16。
        num_stages (int, optional): 下采样的层数。默认值: 2。
        bilinear (bool, optional): 解码器是否使用双线性插值。默认值: True。
        normtype (str, optional): 归一化类型 ('bn' 或 'in')。默认值: 'bn'。
        kernel_size (int, optional): 卷积核大小，必须为奇数。默认值: 3。

    形状:
        输入 x: (B, C_in, H, W)
        输出: list of Tensors。

    Example:
        >>> encoder = UNetEncoder2D(in_channels=1, base_channels=32, num_stages=3, kernel_size=3)
        >>> x = torch.randn(2, 1, 64, 64)
        >>> features = encoder(x)
        >>> print(len(features))
        4
    """
    def __init__(self, in_channels, base_channels=16, num_stages=2, bilinear=True, normtype="bn", kernel_size=3):
        super().__init__()
        self.inc = DoubleConv2D(in_channels, base_channels, normtype=normtype, kernel_size=kernel_size)
        self.down_stages = nn.ModuleList()
        in_ch = base_channels
        for i in range(num_stages):
            out_ch = in_ch * 2
            self.down_stages.append(Down2D(in_ch, out_ch, normtype=normtype, kernel_size=kernel_size))
            in_ch = out_ch

    def forward(self, x):
        features = [self.inc(x)]
        for stage in self.down_stages:
            features.append(stage(features[-1]))
        return features


class UNetEncoder3D(nn.Module):
    """
    三维 U-Net 编码器 (UNet Encoder 3D)。

    负责提取三维体素或时空数据的多尺度特征。由初始 DoubleConv3D 和多个 Down3D 组成。

    Args:
        in_channels (int): 输入通道数。
        base_channels (int, optional): 初始特征通道数。默认值: 16。
        num_stages (int, optional): 下采样的层数。默认值: 2。
        bilinear (bool, optional): 解码器是否使用双线性插值。默认值: True。
        normtype (str, optional): 归一化类型 ('bn' 或 'in')。默认值: 'bn'。
        kernel_size (int, optional): 卷积核大小，必须为奇数。默认值: 3。

    形状:
        输入 x: (B, C_in, D, H, W)
        输出: list of Tensors。

    Example:
        >>> encoder = UNetEncoder3D(in_channels=4, base_channels=16, num_stages=2, kernel_size=3)
        >>> x = torch.randn(1, 4, 32, 32, 32)
        >>> features = encoder(x)
        >>> print(len(features))
        3
    """
    def __init__(self, in_channels, base_channels=16, num_stages=2, bilinear=True, normtype="bn", kernel_size=3):
        super().__init__()
        self.inc = DoubleConv3D(in_channels, base_channels, normtype=normtype, kernel_size=kernel_size)
        self.down_stages = nn.ModuleList()
        in_ch = base_channels
        for i in range(num_stages):
            out_ch = in_ch * 2
            self.down_stages.append(Down3D(in_ch, out_ch, normtype=normtype, kernel_size=kernel_size))
            in_ch = out_ch

    def forward(self, x):
        features = [self.inc(x)]
        for stage in self.down_stages:
            features.append(stage(features[-1]))
        return features