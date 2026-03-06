import torch
import torch.nn as nn
import torch.nn.functional as F

from onescience.modules.utils.wavelet_utils import compl_mul1d, compl_mul2d, compl_mul3d

class WaveletFourierKernel1D(nn.Module):
    """
    一维多小波傅里叶稀疏核 (1D Multiwavelet Fourier Sparse Kernel)。
    (原 sparseKernelFT1d)

    该模块在多小波变换的隐空间中运行。它将输入的 1D 特征转换到傅里叶频域，
    截断高频分量（由 alpha 控制），并对保留的低频模态应用可学习的复数权重矩阵，
    最后通过逆傅里叶变换返回物理空间。这等效于在物理空间进行全局大卷积。

    Args:
        k (int): 多小波基的阶数/块大小。
        alpha (int): 保留的傅里叶模态数量（频率分量数）。
        c (int, optional): 通道缩放因子。默认值: 1。
        nl (int, optional): 保留参数。默认值: 1。
        initializer (callable, optional): 初始化函数（保留接口）。

    形状:
        输入 x: (B, N, c, k)，其中 N 是序列长度。
        输出: (B, N, c, k)，形状与输入一致。

    Example:
        >>> layer = WaveletFourierKernel1D(k=3, alpha=16, c=1)
        >>> x = torch.randn(10, 128, 1, 3)
        >>> out = layer(x)
        >>> print(out.shape)
        torch.Size([10, 128, 1, 3])
    """
    def __init__(self, k, alpha, c=1, nl=1, initializer=None, **kwargs):
        super().__init__()
        self.modes1 = alpha
        self.scale = (1 / (c * k * c * k))
        self.weights1 = nn.Parameter(self.scale * torch.rand(c * k, c * k, self.modes1, dtype=torch.cfloat))
        self.weights1.requires_grad = True
        self.k = k

    def forward(self, x):
        B, N, c, k = x.shape  
        x = x.view(B, N, -1).permute(0, 2, 1)
        x_fft = torch.fft.rfft(x)
        
        l = min(self.modes1, N // 2 + 1)
        out_ft = torch.zeros(B, c * k, N // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :l] = compl_mul1d(x_fft[:, :, :l], self.weights1[:, :, :l])

        x = torch.fft.irfft(out_ft, n=N)
        x = x.permute(0, 2, 1).view(B, N, c, k)
        return x


class WaveletFourierKernel2D(nn.Module):
    """
    二维多小波傅里叶稀疏核 (2D Multiwavelet Fourier Sparse Kernel)。
    (原 sparseKernelFT2d)

    处理 2D 多小波特征。利用 2D FFT 将信号转换到频域，提取四个角的低频模态
    （由于共轭对称性，代码中使用两组权重处理正负频率），进行复数加权聚合后，通过逆变换和线性层返回物理空间。

    Args:
        k (int): 多小波基阶数/块大小。输入最后一个维度应为 k^2。
        alpha (int): 保留的傅里叶模态数量。
        c (int, optional): 通道缩放因子。默认值: 1。
        nl (int, optional): 保留参数。默认值: 1。
        initializer (callable, optional): 初始化函数。

    形状:
        输入 x: (B, N_x, N_y, c, k^2)。
        输出: (B, N_x, N_y, c, k^2)。

    Example:
        >>> layer = WaveletFourierKernel2D(k=3, alpha=8, c=1)
        >>> x = torch.randn(2, 64, 64, 1, 9)
        >>> out = layer(x)
        >>> print(out.shape)
        torch.Size([2, 64, 64, 1, 9])
    """
    def __init__(self, k, alpha, c=1, nl=1, initializer=None, **kwargs):
        super().__init__()
        self.modes = alpha
        self.weights1 = nn.Parameter(torch.zeros(c * k ** 2, c * k ** 2, self.modes, self.modes, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(torch.zeros(c * k ** 2, c * k ** 2, self.modes, self.modes, dtype=torch.cfloat))
        nn.init.xavier_normal_(self.weights1)
        nn.init.xavier_normal_(self.weights2)
        self.Lo = nn.Linear(c * k ** 2, c * k ** 2)
        self.k = k

    def forward(self, x):
        B, Nx, Ny, c, ich = x.shape  
        x = x.view(B, Nx, Ny, -1).permute(0, 3, 1, 2)
        x_fft = torch.fft.rfft2(x)

        l1 = min(self.modes, Nx // 2 + 1)
        l2 = min(self.modes, Ny // 2 + 1)
        out_ft = torch.zeros(B, c * ich, Nx, Ny // 2 + 1, device=x.device, dtype=torch.cfloat)

        out_ft[:, :, :l1, :l2] = compl_mul2d(x_fft[:, :, :l1, :l2], self.weights1[:, :, :l1, :l2])
        out_ft[:, :, -l1:, :l2] = compl_mul2d(x_fft[:, :, -l1:, :l2], self.weights2[:, :, :l1, :l2])

        x = torch.fft.irfft2(out_ft, s=(Nx, Ny)).permute(0, 2, 3, 1)
        x = F.relu(x)
        x = self.Lo(x)
        x = x.view(B, Nx, Ny, c, ich)
        return x


class WaveletFourierKernel3D(nn.Module):
    """
    三维多小波傅里叶稀疏核 (3D Multiwavelet Fourier Sparse Kernel)。
    (原 sparseKernelFT3d)

    在三维频域内操作（适用于 3D 空间或 2D+Time 时空数据）。使用 rfftn 计算三维频谱，
    针对频谱的关键低频部分使用 4 组复数权重进行特征交互，最后还原回物理空间。能有效捕捉体数据中的全局动态模式。

    Args:
        k (int): 多小波参数。输入最后维度应为 k^2。
        alpha (int): 傅里叶模态数。
        c (int, optional): 通道因子。默认值: 1。
        nl (int, optional): 保留参数。默认值: 1。
        initializer (callable, optional): 初始化函数。

    形状:
        输入 x: (B, N_x, N_y, T, c, k^2)。
        输出: (B, N_x, N_y, T, c, k^2)。

    Example:
        >>> layer = WaveletFourierKernel3D(k=3, alpha=8, c=1)
        >>> x = torch.randn(2, 16, 16, 16, 1, 9)
        >>> out = layer(x)
        >>> print(out.shape)
        torch.Size([2, 16, 16, 16, 1, 9])
    """
    def __init__(self, k, alpha, c=1, nl=1, initializer=None, **kwargs):
        super().__init__()
        self.modes = alpha
        self.weights1 = nn.Parameter(torch.zeros(c * k ** 2, c * k ** 2, self.modes, self.modes, self.modes, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(torch.zeros(c * k ** 2, c * k ** 2, self.modes, self.modes, self.modes, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(torch.zeros(c * k ** 2, c * k ** 2, self.modes, self.modes, self.modes, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(torch.zeros(c * k ** 2, c * k ** 2, self.modes, self.modes, self.modes, dtype=torch.cfloat))
        
        nn.init.xavier_normal_(self.weights1)
        nn.init.xavier_normal_(self.weights2)
        nn.init.xavier_normal_(self.weights3)
        nn.init.xavier_normal_(self.weights4)

        self.Lo = nn.Linear(c * k ** 2, c * k ** 2)
        self.k = k

    def forward(self, x):
        B, Nx, Ny, T, c, ich = x.shape  
        x = x.view(B, Nx, Ny, T, -1).permute(0, 4, 1, 2, 3)
        x_fft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        l1 = min(self.modes, Nx // 2 + 1)
        l2 = min(self.modes, Ny // 2 + 1)
        out_ft = torch.zeros(B, c * ich, Nx, Ny, T // 2 + 1, device=x.device, dtype=torch.cfloat)

        out_ft[:, :, :l1, :l2, :self.modes] = compl_mul3d(x_fft[:, :, :l1, :l2, :self.modes], self.weights1[:, :, :l1, :l2, :])
        out_ft[:, :, -l1:, :l2, :self.modes] = compl_mul3d(x_fft[:, :, -l1:, :l2, :self.modes], self.weights2[:, :, :l1, :l2, :])
        out_ft[:, :, :l1, -l2:, :self.modes] = compl_mul3d(x_fft[:, :, :l1, -l2:, :self.modes], self.weights3[:, :, :l1, :l2, :])
        out_ft[:, :, -l1:, -l2:, :self.modes] = compl_mul3d(x_fft[:, :, -l1:, -l2:, :self.modes], self.weights4[:, :, :l1, :l2, :])

        x = torch.fft.irfftn(out_ft, s=(Nx, Ny, T))
        x = x.permute(0, 2, 3, 4, 1)
        x = F.relu(x)
        x = self.Lo(x)
        x = x.view(B, Nx, Ny, T, c, ich)
        return x