import torch
import torch.nn as nn

class WaveletSpatialKernel2D(nn.Module):
    """
    二维空间稀疏核层 (2D Spatial Sparse Kernel)。
    (原 sparseKernel2d / sparseKernel)

    该模块在物理空间中使用标准的二维卷积（Conv2d）来处理多小波系数。
    它先通过卷积层（包含 ReLU 激活）提取特征，然后通过线性层进行特征投影。
    常用于多小波变换中处理高频细节系数，捕捉局部物理特征。

    Args:
        k (int): 多小波块大小参数。输入特征的最后一个维度应为 k^2。
        alpha (int): 控制卷积层输出通道数的倍率因子。
        c (int, optional): 通道缩放因子。默认值: 1。
        nl (int, optional): 保留参数。默认值: 1。
        initializer (callable, optional): 初始化函数（保留接口）。

    形状:
        输入 x: (B, N_x, N_y, c, k^2)。
        输出: (B, N_x, N_y, c, k^2)。

    Example:
        >>> layer = WaveletSpatialKernel2D(k=3, alpha=4, c=1)
        >>> x = torch.randn(4, 64, 64, 1, 9)
        >>> out = layer(x)
        >>> print(out.shape)
        torch.Size([4, 64, 64, 1, 9])
    """
    def __init__(self, k, alpha, c=1, nl=1, initializer=None, **kwargs):
        super().__init__()
        self.k = k
        self.conv = self.convBlock(k, c * k ** 2, alpha)
        self.Lo = nn.Linear(alpha * k ** 2, c * k ** 2)

    def forward(self, x):
        B, Nx, Ny, c, ich = x.shape  
        x = x.view(B, Nx, Ny, -1).permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.Lo(x)
        x = x.view(B, Nx, Ny, c, ich)
        return x

    def convBlock(self, k, W, alpha):
        och = alpha * k ** 2
        net = nn.Sequential(
            nn.Conv2d(W, och, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        return net


class WaveletSpatialKernel3D(nn.Module):
    """
    三维空间稀疏核层 (3D Spatial Sparse Kernel)。
    (原 sparseKernel3d)

    类似于 WaveletSpatialKernel2D，但在三维物理空间上使用 Conv3d 进行局部特征提取。
    适用于处理 3D 体数据中的局部高频特征，作为多小波变换处理三维细节系数的核心组件。

    Args:
        k (int): 多小波参数。输入最后维度应为 k^2。
        alpha (int): 通道倍率因子。
        c (int, optional): 通道因子。默认值: 1。
        nl (int, optional): 保留参数。默认值: 1。
        initializer (callable, optional): 初始化函数。

    形状:
        输入 x: (B, N_x, N_y, T, c, k^2)。通常用于 2D 空间 + 1D 时间，或者 3D 空间。
        输出: (B, N_x, N_y, T, c, k^2)。

    Example:
        >>> layer = WaveletSpatialKernel3D(k=3, alpha=4, c=1)
        >>> x = torch.randn(2, 32, 32, 10, 1, 9)
        >>> out = layer(x)
        >>> print(out.shape)
        torch.Size([2, 32, 32, 10, 1, 9])
    """
    def __init__(self, k, alpha, c=1, nl=1, initializer=None, **kwargs):
        super().__init__()
        self.k = k
        self.conv = self.convBlock(alpha * k ** 2, alpha * k ** 2)
        self.Lo = nn.Linear(alpha * k ** 2, c * k ** 2)

    def forward(self, x):
        B, Nx, Ny, T, c, ich = x.shape  
        x = x.view(B, Nx, Ny, T, -1).permute(0, 4, 1, 2, 3)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.Lo(x)
        x = x.view(B, Nx, Ny, T, c, ich)
        return x

    def convBlock(self, ich, och):
        net = nn.Sequential(
            nn.Conv3d(och, och, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        return net