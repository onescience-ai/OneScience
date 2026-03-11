import torch
import torch.nn as nn
import numpy as np
import math
from typing import List
from torch import Tensor

# --- 导入所需工具与核函数 ---
from onescience.modules.utils.wavelet_utils import get_filter
from .WaveletFourierKernel import WaveletFourierKernel1D, WaveletFourierKernel2D, WaveletFourierKernel3D
from .WaveletSpatialKernel import WaveletSpatialKernel2D, WaveletSpatialKernel3D

class MultiWaveletTransform1D(nn.Module):
    """
    一维多小波变换层 (1D Multiwavelet Transform Layer)。

    该模块实现了基于正交多小波变换的非标准分解与重构。它将 1D 信号分解为不同尺度的近似和细节系数。
    在每个尺度上，分别使用 WaveletFourierKernel1D 作为稀疏核对系数进行特征交互和映射，
    有效结合了多尺度局部性与频域的长程依赖捕捉能力。

    Args:
        k (int, optional): 多小波基的阶数/块大小。默认值: 3。
        alpha (int, optional): 稀疏核中的模态数量控制参数。默认值: 5。
        L (int, optional): 保持不分解的最粗糙层级数。默认值: 0。
        c (int, optional): 通道缩放因子。默认值: 1。
        base (str, optional): 多小波基类型 (如 'legendre' 或 'chebyshev')。默认值: 'legendre'。
        initializer (callable, optional): 参数初始化函数。

    形状:
        输入 x: (B, N, c, k)，其中 N 通常需要是 2 的幂。
        输出: (B, N, c, k)。

    Example:
        >>> model = MultiWaveletTransform1D(k=3, alpha=8, L=0, c=1)
        >>> x = torch.randn(8, 256, 1, 3)
        >>> out = model(x)
        >>> print(out.shape)
        torch.Size([8, 256, 1, 3])
    """
    def __init__(self, k=3, alpha=5, L=0, c=1, base='legendre', initializer=None, **kwargs):
        super().__init__()
        self.k = k
        self.L = L
        H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)
        
        H0r, G0r = H0 @ PHI0, G0 @ PHI0
        H1r, G1r = H1 @ PHI1, G1 @ PHI1
        H0r[np.abs(H0r) < 1e-8] = 0
        H1r[np.abs(H1r) < 1e-8] = 0
        G0r[np.abs(G0r) < 1e-8] = 0
        G1r[np.abs(G1r) < 1e-8] = 0

        self.A = WaveletFourierKernel1D(k, alpha, c)
        self.B = WaveletFourierKernel1D(k, alpha, c)
        self.C = WaveletFourierKernel1D(k, alpha, c)
        self.T0 = nn.Linear(k, k)

        self.register_buffer('ec_s', torch.Tensor(np.concatenate((H0.T, H1.T), axis=0)))
        self.register_buffer('ec_d', torch.Tensor(np.concatenate((G0.T, G1.T), axis=0)))
        self.register_buffer('rc_e', torch.Tensor(np.concatenate((H0r, G0r), axis=0)))
        self.register_buffer('rc_o', torch.Tensor(np.concatenate((H1r, G1r), axis=0)))

    def forward(self, x):
        B, N, c, ich = x.shape  
        ns = math.floor(np.log2(N))

        Ud = torch.jit.annotate(List[Tensor], [])
        Us = torch.jit.annotate(List[Tensor], [])
        
        # decompose
        for i in range(ns - self.L):
            d, x = self.wavelet_transform(x)
            Ud += [self.A(d) + self.B(x)]
            Us += [self.C(d)]
        x = self.T0(x)  # coarsest scale transform

        # reconstruct
        for i in range(ns - 1 - self.L, -1, -1):
            x = x + Us[i]
            x = torch.cat((x, Ud[i]), -1)
            x = self.evenOdd(x)
        return x

    def wavelet_transform(self, x):
        xa = torch.cat([x[:, ::2, :, :], x[:, 1::2, :, :]], -1)
        d = torch.matmul(xa, self.ec_d)
        s = torch.matmul(xa, self.ec_s)
        return d, s

    def evenOdd(self, x):
        B, N, c, ich = x.shape  
        assert ich == 2 * self.k
        x_e = torch.matmul(x, self.rc_e)
        x_o = torch.matmul(x, self.rc_o)

        x_out = torch.zeros(B, N * 2, c, self.k, device=x.device)
        x_out[..., ::2, :, :] = x_e
        x_out[..., 1::2, :, :] = x_o
        return x_out


class MultiWaveletTransform2D(nn.Module):
    """
    二维多小波变换层 (2D Multiwavelet Transform Layer)。
    (原 MWT_CZ2d)

    

    该模块实现了二维正交多小波的非标准分解与重构。它将 2D 网格数据分解为多个尺度，
    在每一层级，使用 WaveletFourierKernel2D (频域核) 处理近似信息 (A 分支)，
    使用 WaveletSpatialKernel2D (空域卷积核) 处理高频细节信息 (B, C 分支)。
    这种混合架构有效地兼顾了频域的大感受野与空域的局部特征敏感性。

    Args:
        k (int, optional): 多小波基阶数。默认值: 3。
        alpha (int, optional): 核函数模态数或通道倍率。默认值: 5。
        L (int, optional): 不进行分解的最粗糙层级数。默认值: 0。
        c (int, optional): 通道缩放因子。默认值: 1。
        base (str, optional): 多小波基类型。默认值: 'legendre'。
        initializer (callable, optional): 初始化函数。

    形状:
        输入 x: (B, N_x, N_y, c, k^2)。Nx 和 Ny 必须是 2 的幂。
        输出: (B, N_x, N_y, c, k^2)。

    Example:
        >>> model = MultiWaveletTransform2D(k=3, alpha=8, L=0, c=1)
        >>> x = torch.randn(2, 64, 64, 1, 9)
        >>> out = model(x)
        >>> print(out.shape)
        torch.Size([2, 64, 64, 1, 9])
    """
    def __init__(self, k=3, alpha=5, L=0, c=1, base='legendre', initializer=None, **kwargs):
        super().__init__()
        self.k = k
        self.L = L
        H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)
        
        H0r, G0r = H0 @ PHI0, G0 @ PHI0
        H1r, G1r = H1 @ PHI1, G1 @ PHI1
        H0r[np.abs(H0r) < 1e-8] = 0
        H1r[np.abs(H1r) < 1e-8] = 0
        G0r[np.abs(G0r) < 1e-8] = 0
        G1r[np.abs(G1r) < 1e-8] = 0

        self.A = WaveletFourierKernel2D(k, alpha, c)
        self.B = WaveletSpatialKernel2D(k, c, c)
        self.C = WaveletSpatialKernel2D(k, c, c)
        self.T0 = nn.Linear(c * k ** 2, c * k ** 2)

        if initializer is not None:
            self.reset_parameters(initializer)

        self.register_buffer('ec_s', torch.Tensor(
            np.concatenate((np.kron(H0, H0).T, np.kron(H0, H1).T, np.kron(H1, H0).T, np.kron(H1, H1).T), axis=0)))
        self.register_buffer('ec_d', torch.Tensor(
            np.concatenate((np.kron(G0, G0).T, np.kron(G0, G1).T, np.kron(G1, G0).T, np.kron(G1, G1).T), axis=0)))

        self.register_buffer('rc_ee', torch.Tensor(np.concatenate((np.kron(H0r, H0r), np.kron(G0r, G0r)), axis=0)))
        self.register_buffer('rc_eo', torch.Tensor(np.concatenate((np.kron(H0r, H1r), np.kron(G0r, G1r)), axis=0)))
        self.register_buffer('rc_oe', torch.Tensor(np.concatenate((np.kron(H1r, H0r), np.kron(G1r, G0r)), axis=0)))
        self.register_buffer('rc_oo', torch.Tensor(np.concatenate((np.kron(H1r, H1r), np.kron(G1r, G1r)), axis=0)))

    def forward(self, x):
        B, Nx, Ny, c, ich = x.shape  
        ns = math.floor(np.log2(Nx))

        Ud = torch.jit.annotate(List[Tensor], [])
        Us = torch.jit.annotate(List[Tensor], [])

        # decompose
        for i in range(ns - self.L):
            d, x = self.wavelet_transform(x)
            Ud += [self.A(d) + self.B(x)]
            Us += [self.C(d)]
        x = self.T0(x.view(B, 2 ** self.L, 2 ** self.L, -1)).view(
            B, 2 ** self.L, 2 ** self.L, c, ich)  

        # reconstruct
        for i in range(ns - 1 - self.L, -1, -1):
            x = x + Us[i]
            x = torch.cat((x, Ud[i]), -1)
            x = self.evenOdd(x)
        return x

    def wavelet_transform(self, x):
        xa = torch.cat([x[:, ::2, ::2, :, :], x[:, ::2, 1::2, :, :],
                        x[:, 1::2, ::2, :, :], x[:, 1::2, 1::2, :, :]], -1)
        d = torch.matmul(xa, self.ec_d)
        s = torch.matmul(xa, self.ec_s)
        return d, s

    def evenOdd(self, x):
        B, Nx, Ny, c, ich = x.shape  
        assert ich == 2 * self.k ** 2
        x_ee = torch.matmul(x, self.rc_ee)
        x_eo = torch.matmul(x, self.rc_eo)
        x_oe = torch.matmul(x, self.rc_oe)
        x_oo = torch.matmul(x, self.rc_oo)

        x_out = torch.zeros(B, Nx * 2, Ny * 2, c, self.k ** 2, device=x.device)
        x_out[:, ::2, ::2, :, :] = x_ee
        x_out[:, ::2, 1::2, :, :] = x_eo
        x_out[:, 1::2, ::2, :, :] = x_oe
        x_out[:, 1::2, 1::2, :, :] = x_oo
        return x_out

    def reset_parameters(self, initializer):
        initializer(self.T0.weight)


class MultiWaveletTransform3D(nn.Module):
    """
    三维多小波变换层 (3D Multiwavelet Transform Layer)。
    (原 MWT_CZ3d)

    针对三维数据（如流体模拟的 3D 场或 2D+T 时空序列）进行多尺度分析。
    通过三维小波变换金字塔分解数据，并在每一层尺度上结合 3D 频域核 (WaveletFourierKernel3D) 
    与 3D 空域核 (WaveletSpatialKernel3D) 进行混合特征演化。

    Args:
        k (int, optional): 多小波基阶数。默认值: 3。
        alpha (int, optional): 频域核模态数或空域核膨胀系数。默认值: 5。
        L (int, optional): 粗糙层级保留深度。默认值: 0。
        c (int, optional): 通道缩放因子。默认值: 1。
        base (str, optional): 小波基类型 ('legendre' 或 'chebyshev')。默认值: 'legendre'。
        initializer (callable, optional): 初始化函数。

    形状:
        输入 x: (B, N_x, N_y, T, c, k^2)。Nx 和 Ny 必须为 2 的幂。
        输出: (B, N_x, N_y, T, c, k^2)。

    Example:
        >>> model = MultiWaveletTransform3D(k=3, alpha=4, L=0, c=1)
        >>> x = torch.randn(1, 32, 32, 16, 1, 9)
        >>> out = model(x)
        >>> print(out.shape)
        torch.Size([1, 32, 32, 16, 1, 9])
    """
    def __init__(self, k=3, alpha=5, L=0, c=1, base='legendre', initializer=None, **kwargs):
        super().__init__()
        self.k = k
        self.L = L
        H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)
        
        H0r, G0r = H0 @ PHI0, G0 @ PHI0
        H1r, G1r = H1 @ PHI1, G1 @ PHI1
        H0r[np.abs(H0r) < 1e-8] = 0
        H1r[np.abs(H1r) < 1e-8] = 0
        G0r[np.abs(G0r) < 1e-8] = 0
        G1r[np.abs(G1r) < 1e-8] = 0

        self.A = WaveletFourierKernel3D(k, alpha, c)
        self.B = WaveletSpatialKernel3D(k, c, c)
        self.C = WaveletSpatialKernel3D(k, c, c)
        self.T0 = nn.Linear(c * k ** 2, c * k ** 2)

        if initializer is not None:
            self.reset_parameters(initializer)

        self.register_buffer('ec_s', torch.Tensor(
            np.concatenate((np.kron(H0, H0).T, np.kron(H0, H1).T, np.kron(H1, H0).T, np.kron(H1, H1).T), axis=0)))
        self.register_buffer('ec_d', torch.Tensor(
            np.concatenate((np.kron(G0, G0).T, np.kron(G0, G1).T, np.kron(G1, G0).T, np.kron(G1, G1).T), axis=0)))

        self.register_buffer('rc_ee', torch.Tensor(np.concatenate((np.kron(H0r, H0r), np.kron(G0r, G0r)), axis=0)))
        self.register_buffer('rc_eo', torch.Tensor(np.concatenate((np.kron(H0r, H1r), np.kron(G0r, G1r)), axis=0)))
        self.register_buffer('rc_oe', torch.Tensor(np.concatenate((np.kron(H1r, H0r), np.kron(G1r, G0r)), axis=0)))
        self.register_buffer('rc_oo', torch.Tensor(np.concatenate((np.kron(H1r, H1r), np.kron(G1r, G1r)), axis=0)))

    def forward(self, x):
        B, Nx, Ny, T, c, ich = x.shape  
        ns = math.floor(np.log2(Nx))

        Ud = torch.jit.annotate(List[Tensor], [])
        Us = torch.jit.annotate(List[Tensor], [])

        # decompose
        for i in range(ns - self.L):
            d, x = self.wavelet_transform(x)
            Ud += [self.A(d) + self.B(x)]
            Us += [self.C(d)]
        x = self.T0(x.view(B, 2 ** self.L, 2 ** self.L, T, -1)).view(
            B, 2 ** self.L, 2 ** self.L, T, c, ich)  

        # reconstruct
        for i in range(ns - 1 - self.L, -1, -1):
            x = x + Us[i]
            x = torch.cat((x, Ud[i]), -1)
            x = self.evenOdd(x)

        return x

    def wavelet_transform(self, x):
        xa = torch.cat([x[:, ::2, ::2, :, :, :], x[:, ::2, 1::2, :, :, :],
                        x[:, 1::2, ::2, :, :, :], x[:, 1::2, 1::2, :, :, :]], -1)
        d = torch.matmul(xa, self.ec_d)
        s = torch.matmul(xa, self.ec_s)
        return d, s

    def evenOdd(self, x):
        B, Nx, Ny, T, c, ich = x.shape  
        assert ich == 2 * self.k ** 2
        x_ee = torch.matmul(x, self.rc_ee)
        x_eo = torch.matmul(x, self.rc_eo)
        x_oe = torch.matmul(x, self.rc_oe)
        x_oo = torch.matmul(x, self.rc_oo)

        x_out = torch.zeros(B, Nx * 2, Ny * 2, T, c, self.k ** 2, device=x.device)
        x_out[:, ::2, ::2, :, :, :] = x_ee
        x_out[:, ::2, 1::2, :, :, :] = x_eo
        x_out[:, 1::2, ::2, :, :, :] = x_oe
        x_out[:, 1::2, 1::2, :, :, :] = x_oo
        return x_out

    def reset_parameters(self, initializer):
        initializer(self.T0.weight)