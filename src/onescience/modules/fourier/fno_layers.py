import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import math

################################################################
#  1d fourier layer
################################################################
class SpectralConv1d(nn.Module):
    """
    一维傅里叶卷积层。

    该层通过快速傅里叶变换（FFT）、频域复数线性变换和逆变换实现全局卷积操作。
    其核心思想是在频域中对低频模态进行线性变换，从而有效地捕捉序列数据中的全局特征和长程依赖。

    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        modes1 (int): 截断的傅里叶模态数量（频率分量数）。该层仅保留最低的 `modes1` 个频率分量进行计算。
                      注意：`modes1` 最多为输入长度的 `floor(N/2) + 1`。

    形状:
        输入 x: (B, Cin, L)，其中 B 是批量大小，Cin 是输入通道数，L 是序列长度
        输出: (B, Cout, L)，其中 Cout 是输出通道数

    Example:
        >>> # 假设序列长度 L=100，保留低频模态数 16
        >>> spec_conv1d = SpectralConv1d(in_channels=64, out_channels=32, modes1=16)
        >>> x = torch.randn(20, 64, 100)
        >>> out = spec_conv1d(x)
        >>> out.shape
        torch.Size([20, 32, 100])
    """
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


################################################################
# 2d fourier layer
################################################################
class SpectralConv2d(nn.Module):
    """
    二维傅里叶卷积层。

    该层实现了傅里叶神经算子中的二维谱卷积操作。它在二维频域内对输入进行处理。
    为了保持计算效率并捕捉主要特征，该层只对频域矩阵角上的低频分量进行线性变换（复数加权计算）。
    通常用于处理图像、流体切片或二维网格数据。

    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        modes1 (int): 第一个空间维度（高度）上保留的傅里叶模态数量
        modes2 (int): 第二个空间维度（宽度）上保留的傅里叶模态数量

    形状:
        输入 x: (B, Cin, H, W)，其中 H 和 W 分别是空间维度的高度和宽度
        输出: (B, Cout, H, W)，输出的空间分辨率与输入保持一致

    Example:
        >>> # 假设输入为 64x64 的网格
        >>> spec_conv2d = SpectralConv2d(in_channels=32, out_channels=64, modes1=12, modes2=12)
        >>> x = torch.randn(10, 32, 64, 64)
        >>> out = spec_conv2d(x)
        >>> out.shape
        torch.Size([10, 64, 64, 64])
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


################################################################
# 3d fourier layers
################################################################

class SpectralConv3d(nn.Module):
    """
    三维傅里叶卷积层。

    针对三维体数据或时空数据（例如 2D 空间 + 1D 时间），在三维频域的四个角（低频区域）进行张量收缩计算。
    算法流程包括：对最后三个维度进行 rfftn，在三维频域的不同象限分别进行复数权重收缩，最后通过 irfftn 还原。

    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        modes1 (int): 第一维度上保留的傅里叶模态数量
        modes2 (int): 第二维度上保留的傅里叶模态数量
        modes3 (int): 第三维度上保留的傅里叶模态数量

    形状:
        输入 x: (B, Cin, D, H, W)，通常对应 (Batch, Channel, X, Y, Z) 或 (Batch, Channel, X, Y, Time)
        输出: (B, Cout, D, H, W)，输出尺寸与输入尺寸相同

    Example:
        >>> # 假设输入尺寸为 32x32x32
        >>> spec_conv3d = SpectralConv3d(in_channels=4, out_channels=8, modes1=8, modes2=8, modes3=8)
        >>> x = torch.randn(2, 4, 32, 32, 32)
        >>> out = spec_conv3d(x)
        >>> out.shape
        torch.Size([2, 8, 32, 32, 32])
    """
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x
