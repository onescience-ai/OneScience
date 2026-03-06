import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import math
from functools import partial
from scipy.special import eval_legendre
from sympy import Poly, legendre, Symbol, chebyshevt
from torch import Tensor
from typing import List, Tuple

def legendreDer(k, x):
    def _legendre(k, x):
        return (2 * k + 1) * eval_legendre(k, x)

    out = 0
    for i in np.arange(k - 1, -1, -2):
        out += _legendre(i, x)
    return out


def phi_(phi_c, x, lb=0, ub=1):
    mask = np.logical_or(x < lb, x > ub) * 1.0
    return np.polynomial.polynomial.Polynomial(phi_c)(x) * (1 - mask)


def get_phi_psi(k, base):
    x = Symbol('x')
    phi_coeff = np.zeros((k, k))
    phi_2x_coeff = np.zeros((k, k))
    if base == 'legendre':
        for ki in range(k):
            coeff_ = Poly(legendre(ki, 2 * x - 1), x).all_coeffs()
            phi_coeff[ki, :ki + 1] = np.flip(np.sqrt(2 * ki + 1) * np.array(coeff_).astype(np.float64))
            coeff_ = Poly(legendre(ki, 4 * x - 1), x).all_coeffs()
            phi_2x_coeff[ki, :ki + 1] = np.flip(np.sqrt(2) * np.sqrt(2 * ki + 1) * np.array(coeff_).astype(np.float64))

        psi1_coeff = np.zeros((k, k))
        psi2_coeff = np.zeros((k, k))
        for ki in range(k):
            psi1_coeff[ki, :] = phi_2x_coeff[ki, :]
            for i in range(k):
                a = phi_2x_coeff[ki, :ki + 1]
                b = phi_coeff[i, :i + 1]
                prod_ = np.convolve(a, b)
                prod_[np.abs(prod_) < 1e-8] = 0
                proj_ = (prod_ * 1 / (np.arange(len(prod_)) + 1) * np.power(0.5, 1 + np.arange(len(prod_)))).sum()
                psi1_coeff[ki, :] -= proj_ * phi_coeff[i, :]
                psi2_coeff[ki, :] -= proj_ * phi_coeff[i, :]
            for j in range(ki):
                a = phi_2x_coeff[ki, :ki + 1]
                b = psi1_coeff[j, :]
                prod_ = np.convolve(a, b)
                prod_[np.abs(prod_) < 1e-8] = 0
                proj_ = (prod_ * 1 / (np.arange(len(prod_)) + 1) * np.power(0.5, 1 + np.arange(len(prod_)))).sum()
                psi1_coeff[ki, :] -= proj_ * psi1_coeff[j, :]
                psi2_coeff[ki, :] -= proj_ * psi2_coeff[j, :]

            a = psi1_coeff[ki, :]
            prod_ = np.convolve(a, a)
            prod_[np.abs(prod_) < 1e-8] = 0
            norm1 = (prod_ * 1 / (np.arange(len(prod_)) + 1) * np.power(0.5, 1 + np.arange(len(prod_)))).sum()

            a = psi2_coeff[ki, :]
            prod_ = np.convolve(a, a)
            prod_[np.abs(prod_) < 1e-8] = 0
            norm2 = (prod_ * 1 / (np.arange(len(prod_)) + 1) * (1 - np.power(0.5, 1 + np.arange(len(prod_))))).sum()
            norm_ = np.sqrt(norm1 + norm2)
            psi1_coeff[ki, :] /= norm_
            psi2_coeff[ki, :] /= norm_
            psi1_coeff[np.abs(psi1_coeff) < 1e-8] = 0
            psi2_coeff[np.abs(psi2_coeff) < 1e-8] = 0

        phi = [np.poly1d(np.flip(phi_coeff[i, :])) for i in range(k)]
        psi1 = [np.poly1d(np.flip(psi1_coeff[i, :])) for i in range(k)]
        psi2 = [np.poly1d(np.flip(psi2_coeff[i, :])) for i in range(k)]

    elif base == 'chebyshev':
        for ki in range(k):
            if ki == 0:
                phi_coeff[ki, :ki + 1] = np.sqrt(2 / np.pi)
                phi_2x_coeff[ki, :ki + 1] = np.sqrt(2 / np.pi) * np.sqrt(2)
            else:
                coeff_ = Poly(chebyshevt(ki, 2 * x - 1), x).all_coeffs()
                phi_coeff[ki, :ki + 1] = np.flip(2 / np.sqrt(np.pi) * np.array(coeff_).astype(np.float64))
                coeff_ = Poly(chebyshevt(ki, 4 * x - 1), x).all_coeffs()
                phi_2x_coeff[ki, :ki + 1] = np.flip(
                    np.sqrt(2) * 2 / np.sqrt(np.pi) * np.array(coeff_).astype(np.float64))

        phi = [partial(phi_, phi_coeff[i, :]) for i in range(k)]

        x = Symbol('x')
        kUse = 2 * k
        roots = Poly(chebyshevt(kUse, 2 * x - 1)).all_roots()
        x_m = np.array([rt.evalf(20) for rt in roots]).astype(np.float64)
        # x_m[x_m==0.5] = 0.5 + 1e-8 # add small noise to avoid the case of 0.5 belonging to both phi(2x) and phi(2x-1)
        # not needed for our purpose here, we use even k always to avoid
        wm = np.pi / kUse / 2

        psi1_coeff = np.zeros((k, k))
        psi2_coeff = np.zeros((k, k))

        psi1 = [[] for _ in range(k)]
        psi2 = [[] for _ in range(k)]

        for ki in range(k):
            psi1_coeff[ki, :] = phi_2x_coeff[ki, :]
            for i in range(k):
                proj_ = (wm * phi[i](x_m) * np.sqrt(2) * phi[ki](2 * x_m)).sum()
                psi1_coeff[ki, :] -= proj_ * phi_coeff[i, :]
                psi2_coeff[ki, :] -= proj_ * phi_coeff[i, :]

            for j in range(ki):
                proj_ = (wm * psi1[j](x_m) * np.sqrt(2) * phi[ki](2 * x_m)).sum()
                psi1_coeff[ki, :] -= proj_ * psi1_coeff[j, :]
                psi2_coeff[ki, :] -= proj_ * psi2_coeff[j, :]

            psi1[ki] = partial(phi_, psi1_coeff[ki, :], lb=0, ub=0.5)
            psi2[ki] = partial(phi_, psi2_coeff[ki, :], lb=0.5, ub=1)

            norm1 = (wm * psi1[ki](x_m) * psi1[ki](x_m)).sum()
            norm2 = (wm * psi2[ki](x_m) * psi2[ki](x_m)).sum()

            norm_ = np.sqrt(norm1 + norm2)
            psi1_coeff[ki, :] /= norm_
            psi2_coeff[ki, :] /= norm_
            psi1_coeff[np.abs(psi1_coeff) < 1e-8] = 0
            psi2_coeff[np.abs(psi2_coeff) < 1e-8] = 0

            psi1[ki] = partial(phi_, psi1_coeff[ki, :], lb=0, ub=0.5 + 1e-16)
            psi2[ki] = partial(phi_, psi2_coeff[ki, :], lb=0.5 + 1e-16, ub=1)

    return phi, psi1, psi2


def get_filter(base, k):
    def psi(psi1, psi2, i, inp):
        mask = (inp <= 0.5) * 1.0
        return psi1[i](inp) * mask + psi2[i](inp) * (1 - mask)

    if base not in ['legendre', 'chebyshev']:
        raise Exception('Base not supported')

    x = Symbol('x')
    H0 = np.zeros((k, k))
    H1 = np.zeros((k, k))
    G0 = np.zeros((k, k))
    G1 = np.zeros((k, k))
    PHI0 = np.zeros((k, k))
    PHI1 = np.zeros((k, k))
    phi, psi1, psi2 = get_phi_psi(k, base)
    if base == 'legendre':
        roots = Poly(legendre(k, 2 * x - 1)).all_roots()
        x_m = np.array([rt.evalf(20) for rt in roots]).astype(np.float64)
        wm = 1 / k / legendreDer(k, 2 * x_m - 1) / eval_legendre(k - 1, 2 * x_m - 1)

        for ki in range(k):
            for kpi in range(k):
                H0[ki, kpi] = 1 / np.sqrt(2) * (wm * phi[ki](x_m / 2) * phi[kpi](x_m)).sum()
                G0[ki, kpi] = 1 / np.sqrt(2) * (wm * psi(psi1, psi2, ki, x_m / 2) * phi[kpi](x_m)).sum()
                H1[ki, kpi] = 1 / np.sqrt(2) * (wm * phi[ki]((x_m + 1) / 2) * phi[kpi](x_m)).sum()
                G1[ki, kpi] = 1 / np.sqrt(2) * (wm * psi(psi1, psi2, ki, (x_m + 1) / 2) * phi[kpi](x_m)).sum()

        PHI0 = np.eye(k)
        PHI1 = np.eye(k)

    elif base == 'chebyshev':
        x = Symbol('x')
        kUse = 2 * k
        roots = Poly(chebyshevt(kUse, 2 * x - 1)).all_roots()
        x_m = np.array([rt.evalf(20) for rt in roots]).astype(np.float64)
        # x_m[x_m==0.5] = 0.5 + 1e-8 # add small noise to avoid the case of 0.5 belonging to both phi(2x) and phi(2x-1)
        # not needed for our purpose here, we use even k always to avoid
        wm = np.pi / kUse / 2

        for ki in range(k):
            for kpi in range(k):
                H0[ki, kpi] = 1 / np.sqrt(2) * (wm * phi[ki](x_m / 2) * phi[kpi](x_m)).sum()
                G0[ki, kpi] = 1 / np.sqrt(2) * (wm * psi(psi1, psi2, ki, x_m / 2) * phi[kpi](x_m)).sum()
                H1[ki, kpi] = 1 / np.sqrt(2) * (wm * phi[ki]((x_m + 1) / 2) * phi[kpi](x_m)).sum()
                G1[ki, kpi] = 1 / np.sqrt(2) * (wm * psi(psi1, psi2, ki, (x_m + 1) / 2) * phi[kpi](x_m)).sum()

                PHI0[ki, kpi] = (wm * phi[ki](2 * x_m) * phi[kpi](2 * x_m)).sum() * 2
                PHI1[ki, kpi] = (wm * phi[ki](2 * x_m - 1) * phi[kpi](2 * x_m - 1)).sum() * 2

        PHI0[np.abs(PHI0) < 1e-8] = 0
        PHI1[np.abs(PHI1) < 1e-8] = 0

    H0[np.abs(H0) < 1e-8] = 0
    H1[np.abs(H1) < 1e-8] = 0
    G0[np.abs(G0) < 1e-8] = 0
    G1[np.abs(G1) < 1e-8] = 0

    return H0, H1, G0, G1, PHI0, PHI1


def compl_mul1d(x, weights):
    # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
    return torch.einsum("bix,iox->box", x, weights)


class sparseKernelFT1d(nn.Module):
    """
        一维傅里叶稀疏核层 (1D Fourier Sparse Kernel)。

        该模块在频域中执行线性变换。它首先对输入进行一维实数快速傅里叶变换（RFFT），然后对低频模态（Modes）进行复数权重乘法，最后通过逆变换（IRFFT）还原回物理空间。
        这相当于在物理空间中进行全局卷积，能够有效捕捉长程依赖。

        Args:
            k (int): 多小波的元素数量或块大小。
            alpha (int): 保留的傅里叶模态数量（频率分量数）。
            c (int, optional): 通道缩放因子。
            nl (int, optional): 未在代码中使用，保留参数。
            initializer (callable, optional): 初始化函数（未在代码中使用）。

        形状:
            输入: (B, N, c, k)，其中 N 是序列长度。
            输出: (B, N, c, k)。

        Example:
            >>> # 假设多小波块大小 k=3
            >>> layer = sparseKernelFT1d(k=3, alpha=16, c=1)
            >>> x = torch.randn(10, 128, 1, 3)
            >>> out = layer(x)
            >>> out.shape
            torch.Size([10, 128, 1, 3])
    """
    def __init__(self,
                 k, alpha, c=1,
                 nl=1,
                 initializer=None,
                 **kwargs):
        super(sparseKernelFT1d, self).__init__()

        self.modes1 = alpha
        self.scale = (1 / (c * k * c * k))
        self.weights1 = nn.Parameter(self.scale * torch.rand(c * k, c * k, self.modes1, dtype=torch.cfloat))
        self.weights1.requires_grad = True
        self.k = k

    def forward(self, x):
        B, N, c, k = x.shape  # (B, N, c, k)

        x = x.view(B, N, -1)
        x = x.permute(0, 2, 1)
        x_fft = torch.fft.rfft(x)
        # Multiply relevant Fourier modes
        l = min(self.modes1, N // 2 + 1)
        out_ft = torch.zeros(B, c * k, N // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :l] = compl_mul1d(x_fft[:, :, :l], self.weights1[:, :, :l])

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=N)
        x = x.permute(0, 2, 1).view(B, N, c, k)
        return x


class MWT_CZ1d(nn.Module):
    """
        一维多小波变换层 (1D Multiwavelet Transform Layer)。

        该模块实现了基于多小波变换的非标准形式（Non-standard form）分解和重构。它将输入信号分解为不同尺度的近似系数和细节系数。
        在每个尺度上，分别使用稀疏核（A, B, C）对系数进行处理，最后进行重构。这种分层结构有助于捕捉多尺度的物理特征。

        Args:
            k (int, optional): 多小波基的阶数/块大小。默认值: 3。
            alpha (int, optional): 稀疏核中的模态数量或中间通道控制参数。默认值: 5。
            L (int, optional): 保持不分解的最粗糙层级数。变换将进行到 log_2(N) - L 层。默认值: 0。
            c (int, optional): 通道缩放因子。默认值: 1。
            base (str, optional): 多小波基的类型，例如 'legendre'。需要外部 get_filter 函数支持。默认值: 'legendre'。
            initializer (callable, optional): 参数初始化函数。

        形状:
            输入: (B, N, c, k)，其中 N 通常需要是 2 的幂。
            输出: (B, N, c, k)。

        Example:
            >>> model = MWT_CZ1d(k=3, alpha=8, L=0, c=1)
            >>> x = torch.randn(8, 256, 1, 3)
            >>> out = model(x)
            >>> out.shape
            torch.Size([8, 256, 1, 3])
    """
    def __init__(self,
                 k=3, alpha=5,
                 L=0, c=1,
                 base='legendre',
                 initializer=None,
                 **kwargs):
        super(MWT_CZ1d, self).__init__()

        self.k = k
        self.L = L
        H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)
        H0r = H0 @ PHI0
        G0r = G0 @ PHI0
        H1r = H1 @ PHI1
        G1r = G1 @ PHI1

        H0r[np.abs(H0r) < 1e-8] = 0
        H1r[np.abs(H1r) < 1e-8] = 0
        G0r[np.abs(G0r) < 1e-8] = 0
        G1r[np.abs(G1r) < 1e-8] = 0

        self.A = sparseKernelFT1d(k, alpha, c)
        self.B = sparseKernelFT1d(k, alpha, c)
        self.C = sparseKernelFT1d(k, alpha, c)

        self.T0 = nn.Linear(k, k)

        self.register_buffer('ec_s', torch.Tensor(
            np.concatenate((H0.T, H1.T), axis=0)))
        self.register_buffer('ec_d', torch.Tensor(
            np.concatenate((G0.T, G1.T), axis=0)))

        self.register_buffer('rc_e', torch.Tensor(
            np.concatenate((H0r, G0r), axis=0)))
        self.register_buffer('rc_o', torch.Tensor(
            np.concatenate((H1r, G1r), axis=0)))

    def forward(self, x):

        B, N, c, ich = x.shape  # (B, N, k)
        ns = math.floor(np.log2(N))

        Ud = torch.jit.annotate(List[Tensor], [])
        Us = torch.jit.annotate(List[Tensor], [])
        #         decompose
        for i in range(ns - self.L):
            d, x = self.wavelet_transform(x)
            Ud += [self.A(d) + self.B(x)]
            Us += [self.C(d)]
        x = self.T0(x)  # coarsest scale transform

        #        reconstruct
        for i in range(ns - 1 - self.L, -1, -1):
            x = x + Us[i]
            x = torch.cat((x, Ud[i]), -1)
            x = self.evenOdd(x)
        return x

    def wavelet_transform(self, x):
        xa = torch.cat([x[:, ::2, :, :],
                        x[:, 1::2, :, :],
                        ], -1)
        d = torch.matmul(xa, self.ec_d)
        s = torch.matmul(xa, self.ec_s)
        return d, s

    def evenOdd(self, x):

        B, N, c, ich = x.shape  # (B, N, c, k)
        assert ich == 2 * self.k
        x_e = torch.matmul(x, self.rc_e)
        x_o = torch.matmul(x, self.rc_o)

        x = torch.zeros(B, N * 2, c, self.k,
                        device=x.device)
        x[..., ::2, :, :] = x_e
        x[..., 1::2, :, :] = x_o
        return x


class sparseKernel2d(nn.Module):
    """
        二维空间稀疏核层 (2D Spatial Sparse Kernel)。

        该模块在物理空间中使用标准的二维卷积（Conv2d）来处理多小波系数。
        它先通过卷积层（包含 ReLU 激活）提取特征，然后通过线性层进行投影。通常用于处理多小波变换中的高频细节系数或不需要全局傅里叶变换的部分。

        Args:
            k (int): 多小波块大小参数。输入特征的最后一个维度应为 k^2。
            alpha (int): 控制卷积层输出通道数的倍率因子。
            c (int, optional): 通道缩放因子。

        形状:
            输入: (B, N_x, N_y, c, k^2)。
            输出: (B, N_x, N_y, c, k^2)。

        Example:
            >>> # k=3, 输入最后维度为 3*3=9
            >>> layer = sparseKernel2d(k=3, alpha=4, c=1)
            >>> x = torch.randn(4, 64, 64, 1, 9)
            >>> out = layer(x)
            >>> out.shape
            torch.Size([4, 64, 64, 1, 9])
    """
    def __init__(self,
                 k, alpha, c=1,
                 nl=1,
                 initializer=None,
                 **kwargs):
        super(sparseKernel2d, self).__init__()

        self.k = k
        self.conv = self.convBlock(k, c * k ** 2, alpha)
        self.Lo = nn.Linear(alpha * k ** 2, c * k ** 2)

    def forward(self, x):
        B, Nx, Ny, c, ich = x.shape  # (B, Nx, Ny, c, k**2)
        x = x.view(B, Nx, Ny, -1)
        x = x.permute(0, 3, 1, 2)
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


def compl_mul2d(x, weights):
    # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
    return torch.einsum("bixy,ioxy->boxy", x, weights)


class sparseKernelFT2d(nn.Module):
    """
        二维多小波变换层 (2D Multiwavelet Transform Layer)。

        对二维图像或网格数据进行多尺度的多小波分解与重构。
        在每个尺度上，利用 sparseKernelFT2d 处理细节系数（通常对应低频部分或需要全局信息的系数），利用 sparseKernel2d 处理近似系数（或其他高频系数）。
        包含 2D 小波变换（分解为 LL, LH, HL, HH）及相应的重组逻辑，实现了多尺度特征提取。

        Args:
            k (int, optional): 多小波基阶数。默认值: 3。
            alpha (int, optional): 核函数模态数/宽度控制。默认值: 5。
            L (int, optional): 粗糙层级保留数。默认值: 0。
            c (int, optional): 通道因子。默认值: 1。
            base (str, optional): 小波基类型。默认值: 'legendre'。

        形状:
            输入: (B, N_x, N_y, c, k^2)。
            输出: (B, N_x, N_y, c, k^2)。

        Example:
            >>> mwt_2d = MWT_CZ2d(k=3, alpha=8)
            >>> x = torch.randn(2, 64, 64, 1, 9)
            >>> out = mwt_2d(x)
            >>> out.shape
            torch.Size([2, 64, 64, 1, 9])
    """
    def __init__(self,
                 k, alpha, c=1,
                 nl=1,
                 initializer=None,
                 **kwargs):
        super(sparseKernelFT2d, self).__init__()

        self.modes = alpha

        self.weights1 = nn.Parameter(torch.zeros(c * k ** 2, c * k ** 2, self.modes, self.modes, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(torch.zeros(c * k ** 2, c * k ** 2, self.modes, self.modes, dtype=torch.cfloat))
        nn.init.xavier_normal_(self.weights1)
        nn.init.xavier_normal_(self.weights2)

        self.Lo = nn.Linear(c * k ** 2, c * k ** 2)
        self.k = k

    def forward(self, x):
        B, Nx, Ny, c, ich = x.shape  # (B, N, N, c, k^2)

        x = x.view(B, Nx, Ny, -1)
        x = x.permute(0, 3, 1, 2)
        x_fft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        l1 = min(self.modes, Nx // 2 + 1)
        l1l = min(self.modes, Nx // 2 - 1)
        l2 = min(self.modes, Ny // 2 + 1)
        out_ft = torch.zeros(B, c * ich, Nx, Ny // 2 + 1, device=x.device, dtype=torch.cfloat)

        out_ft[:, :, :l1, :l2] = compl_mul2d(
            x_fft[:, :, :l1, :l2], self.weights1[:, :, :l1, :l2])
        out_ft[:, :, -l1:, :l2] = compl_mul2d(
            x_fft[:, :, -l1:, :l2], self.weights2[:, :, :l1, :l2])

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(Nx, Ny))

        x = x.permute(0, 2, 3, 1)
        x = F.relu(x)
        x = self.Lo(x)
        x = x.view(B, Nx, Ny, c, ich)
        return x


class MWT_CZ2d(nn.Module):
    """
        三维空间稀疏核层 (3D Spatial Sparse Kernel)。

        类似于 sparseKernel2d，但在三维空间上使用 Conv3d 进行局部特征提取。
        适用于处理 3D 体数据中的局部高频特征，通常作为多小波变换中处理细节系数的一部分。

        Args:
            k (int): 多小波参数。输入最后维度应为 k^2。
            alpha (int): 通道倍率因子。
            c (int, optional): 通道因子。

        形状:
            输入: (B, N_x, N_y, T, c, k^2)，通常用于 2D 空间 + 1D 时间的数据。
            输出: (B, N_x, N_y, T, c, k^2)。

        Example:
            >>> layer = sparseKernel3d(k=3, alpha=4)
            >>> x = torch.randn(2, 32, 32, 10, 1, 9)
            >>> out = layer(x)
    """
    def __init__(self,
                 k=3, alpha=5,
                 L=0, c=1,
                 base='legendre',
                 initializer=None,
                 **kwargs):
        super(MWT_CZ2d, self).__init__()

        self.k = k
        self.L = L
        H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)
        H0r = H0 @ PHI0
        G0r = G0 @ PHI0
        H1r = H1 @ PHI1
        G1r = G1 @ PHI1
        H0r[np.abs(H0r) < 1e-8] = 0
        H1r[np.abs(H1r) < 1e-8] = 0
        G0r[np.abs(G0r) < 1e-8] = 0
        G1r[np.abs(G1r) < 1e-8] = 0

        self.A = sparseKernelFT2d(k, alpha, c)
        self.B = sparseKernel2d(k, c, c)
        self.C = sparseKernel2d(k, c, c)

        self.T0 = nn.Linear(c * k ** 2, c * k ** 2)

        if initializer is not None:
            self.reset_parameters(initializer)

        self.register_buffer('ec_s', torch.Tensor(
            np.concatenate((np.kron(H0, H0).T,
                            np.kron(H0, H1).T,
                            np.kron(H1, H0).T,
                            np.kron(H1, H1).T,
                            ), axis=0)))
        self.register_buffer('ec_d', torch.Tensor(
            np.concatenate((np.kron(G0, G0).T,
                            np.kron(G0, G1).T,
                            np.kron(G1, G0).T,
                            np.kron(G1, G1).T,
                            ), axis=0)))

        self.register_buffer('rc_ee', torch.Tensor(
            np.concatenate((np.kron(H0r, H0r),
                            np.kron(G0r, G0r),
                            ), axis=0)))
        self.register_buffer('rc_eo', torch.Tensor(
            np.concatenate((np.kron(H0r, H1r),
                            np.kron(G0r, G1r),
                            ), axis=0)))
        self.register_buffer('rc_oe', torch.Tensor(
            np.concatenate((np.kron(H1r, H0r),
                            np.kron(G1r, G0r),
                            ), axis=0)))
        self.register_buffer('rc_oo', torch.Tensor(
            np.concatenate((np.kron(H1r, H1r),
                            np.kron(G1r, G1r),
                            ), axis=0)))

    def forward(self, x):

        B, Nx, Ny, c, ich = x.shape  # (B, Nx, Ny, c, k**2)
        ns = math.floor(np.log2(Nx))

        Ud = torch.jit.annotate(List[Tensor], [])
        Us = torch.jit.annotate(List[Tensor], [])

        # decompose
        for i in range(ns - self.L):
            d, x = self.wavelet_transform(x)
            Ud += [self.A(d) + self.B(x)]
            Us += [self.C(d)]
        x = self.T0(x.view(B, 2 ** self.L, 2 ** self.L, -1)).view(
            B, 2 ** self.L, 2 ** self.L, c, ich)  # coarsest scale transform

        # reconstruct
        for i in range(ns - 1 - self.L, -1, -1):
            x = x + Us[i]
            x = torch.cat((x, Ud[i]), -1)
            x = self.evenOdd(x)

        return x

    def wavelet_transform(self, x):
        xa = torch.cat([x[:, ::2, ::2, :, :],
                        x[:, ::2, 1::2, :, :],
                        x[:, 1::2, ::2, :, :],
                        x[:, 1::2, 1::2, :, :]
                        ], -1)
        d = torch.matmul(xa, self.ec_d)
        s = torch.matmul(xa, self.ec_s)
        return d, s

    def evenOdd(self, x):

        B, Nx, Ny, c, ich = x.shape  # (B, Nx, Ny, c, k**2)
        assert ich == 2 * self.k ** 2
        x_ee = torch.matmul(x, self.rc_ee)
        x_eo = torch.matmul(x, self.rc_eo)
        x_oe = torch.matmul(x, self.rc_oe)
        x_oo = torch.matmul(x, self.rc_oo)

        x = torch.zeros(B, Nx * 2, Ny * 2, c, self.k ** 2,
                        device=x.device)
        x[:, ::2, ::2, :, :] = x_ee
        x[:, ::2, 1::2, :, :] = x_eo
        x[:, 1::2, ::2, :, :] = x_oe
        x[:, 1::2, 1::2, :, :] = x_oo
        return x

    def reset_parameters(self, initializer):
        initializer(self.T0.weight)


class sparseKernel(nn.Module):
    def __init__(self,
                 k, alpha, c=1,
                 nl=1,
                 initializer=None,
                 **kwargs):
        super(sparseKernel, self).__init__()

        self.k = k
        self.conv = self.convBlock(k, c * k ** 2, alpha)
        self.Lo = nn.Linear(alpha * k ** 2, c * k ** 2)

    def forward(self, x):
        B, Nx, Ny, c, ich = x.shape  # (B, Nx, Ny, c, k**2)
        x = x.view(B, Nx, Ny, -1)
        x = x.permute(0, 3, 1, 2)
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


class sparseKernel3d(nn.Module):
    """
        三维傅里叶稀疏核层 (3D Fourier Sparse Kernel)。

        在三维频域内操作。该层使用 rfftn 计算三维频谱，并针对 3D 频谱的 8 个角（通过共轭对称性简化为 4 组权重）进行复数加权处理。
        这允许模型捕捉体数据或时空数据（如流体演化）中的全局模式和周期性特征。

        Args:
            k (int): 多小波参数。
            alpha (int): 傅里叶模态数。
            c (int, optional): 通道因子。

        形状:
            输入: (B, N_x, N_y, T, c, k^2)。
            输出: (B, N_x, N_y, T, c, k^2)。

        Example:
            >>> layer = sparseKernelFT3d(k=3, alpha=8)
            >>> x = torch.randn(2, 16, 16, 16, 1, 9)
            >>> out = layer(x)
    """
    def __init__(self,
                 k, alpha, c=1,
                 nl=1,
                 initializer=None,
                 **kwargs):
        super(sparseKernel3d, self).__init__()

        self.k = k
        self.conv = self.convBlock(alpha * k ** 2, alpha * k ** 2)
        self.Lo = nn.Linear(alpha * k ** 2, c * k ** 2)

    def forward(self, x):
        B, Nx, Ny, T, c, ich = x.shape  # (B, Nx, Ny, T, c, k**2)
        x = x.view(B, Nx, Ny, T, -1)
        x = x.permute(0, 4, 1, 2, 3)
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


def compl_mul3d(input, weights):
    # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    return torch.einsum("bixyz,ioxyz->boxyz", input, weights)


class sparseKernelFT3d(nn.Module):
    """
        三维傅里叶稀疏核层 (3D Fourier Sparse Kernel)。

        在三维频域内操作。该层使用 rfftn 计算三维频谱，并针对 3D 频谱的 8 个角（通过共轭对称性简化为 4 组权重）进行复数加权处理。
        这允许模型捕捉体数据或时空数据（如流体演化）中的全局模式和周期性特征。

        Args:
            k (int): 多小波参数。
            alpha (int): 傅里叶模态数。
            c (int, optional): 通道因子。

        形状:
            输入: (B, N_x, N_y, T, c, k^2)。
            输出: (B, N_x, N_y, T, c, k^2)。

        Example:
            >>> layer = sparseKernelFT3d(k=3, alpha=8)
            >>> x = torch.randn(2, 16, 16, 16, 1, 9)
            >>> out = layer(x)
    """
    def __init__(self,
                 k, alpha, c=1,
                 nl=1,
                 initializer=None,
                 **kwargs):
        super(sparseKernelFT3d, self).__init__()

        self.modes = alpha

        self.weights1 = nn.Parameter(
            torch.zeros(c * k ** 2, c * k ** 2, self.modes, self.modes, self.modes, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            torch.zeros(c * k ** 2, c * k ** 2, self.modes, self.modes, self.modes, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            torch.zeros(c * k ** 2, c * k ** 2, self.modes, self.modes, self.modes, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            torch.zeros(c * k ** 2, c * k ** 2, self.modes, self.modes, self.modes, dtype=torch.cfloat))
        nn.init.xavier_normal_(self.weights1)
        nn.init.xavier_normal_(self.weights2)
        nn.init.xavier_normal_(self.weights3)
        nn.init.xavier_normal_(self.weights4)

        self.Lo = nn.Linear(c * k ** 2, c * k ** 2)
        self.k = k

    def forward(self, x):
        B, Nx, Ny, T, c, ich = x.shape  # (B, N, N, T, c, k^2)

        x = x.view(B, Nx, Ny, T, -1)
        x = x.permute(0, 4, 1, 2, 3)
        x_fft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        l1 = min(self.modes, Nx // 2 + 1)
        l2 = min(self.modes, Ny // 2 + 1)
        out_ft = torch.zeros(B, c * ich, Nx, Ny, T // 2 + 1, device=x.device, dtype=torch.cfloat)

        out_ft[:, :, :l1, :l2, :self.modes] = compl_mul3d(
            x_fft[:, :, :l1, :l2, :self.modes], self.weights1[:, :, :l1, :l2, :])
        out_ft[:, :, -l1:, :l2, :self.modes] = compl_mul3d(
            x_fft[:, :, -l1:, :l2, :self.modes], self.weights2[:, :, :l1, :l2, :])
        out_ft[:, :, :l1, -l2:, :self.modes] = compl_mul3d(
            x_fft[:, :, :l1, -l2:, :self.modes], self.weights3[:, :, :l1, :l2, :])
        out_ft[:, :, -l1:, -l2:, :self.modes] = compl_mul3d(
            x_fft[:, :, -l1:, -l2:, :self.modes], self.weights4[:, :, :l1, :l2, :])

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(Nx, Ny, T))

        x = x.permute(0, 2, 3, 4, 1)
        x = F.relu(x)
        x = self.Lo(x)
        x = x.view(B, Nx, Ny, T, c, ich)
        return x


class MWT_CZ3d(nn.Module):
    """
        三维多小波变换层 (3D Multiwavelet Transform Layer)。

        针对三维数据（如流体模拟的 3D 场或 2D+T 序列）进行多尺度分析。
        通过三维小波变换分解数据，并在不同尺度上应用 3D 稀疏核（包括频域和空域核）进行特征变换。
        这种分层处理使得模型能够同时捕捉局部细节（通过空间核）和全局动态（通过谱核）。

        Args:
            k (int, optional): 多小波基阶数。默认值: 3。
            alpha (int, optional): 核参数。默认值: 5。
            L (int, optional): 粗糙层级。默认值: 0。
            c (int, optional): 通道因子。默认值: 1。
            base (str, optional): 小波基类型。默认值: 'legendre'。

        形状:
            输入: (B, N_x, N_y, T, c, k^2)。
            输出: (B, N_x, N_y, T, c, k^2)。

        Example:
            >>> mwt_3d = MWT_CZ3d(k=3, alpha=4)
            >>> x = torch.randn(1, 32, 32, 32, 1, 9)
            >>> out = mwt_3d(x)
            >>> out.shape
            torch.Size([1, 32, 32, 32, 1, 9])
    """
    def __init__(self,
                 k=3, alpha=5,
                 L=0, c=1,
                 base='legendre',
                 initializer=None,
                 **kwargs):
        super(MWT_CZ3d, self).__init__()

        self.k = k
        self.L = L
        H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)
        H0r = H0 @ PHI0
        G0r = G0 @ PHI0
        H1r = H1 @ PHI1
        G1r = G1 @ PHI1

        H0r[np.abs(H0r) < 1e-8] = 0
        H1r[np.abs(H1r) < 1e-8] = 0
        G0r[np.abs(G0r) < 1e-8] = 0
        G1r[np.abs(G1r) < 1e-8] = 0

        self.A = sparseKernelFT3d(k, alpha, c)
        self.B = sparseKernel3d(k, c, c)
        self.C = sparseKernel3d(k, c, c)

        self.T0 = nn.Linear(c * k ** 2, c * k ** 2)

        if initializer is not None:
            self.reset_parameters(initializer)

        self.register_buffer('ec_s', torch.Tensor(
            np.concatenate((np.kron(H0, H0).T,
                            np.kron(H0, H1).T,
                            np.kron(H1, H0).T,
                            np.kron(H1, H1).T,
                            ), axis=0)))
        self.register_buffer('ec_d', torch.Tensor(
            np.concatenate((np.kron(G0, G0).T,
                            np.kron(G0, G1).T,
                            np.kron(G1, G0).T,
                            np.kron(G1, G1).T,
                            ), axis=0)))

        self.register_buffer('rc_ee', torch.Tensor(
            np.concatenate((np.kron(H0r, H0r),
                            np.kron(G0r, G0r),
                            ), axis=0)))
        self.register_buffer('rc_eo', torch.Tensor(
            np.concatenate((np.kron(H0r, H1r),
                            np.kron(G0r, G1r),
                            ), axis=0)))
        self.register_buffer('rc_oe', torch.Tensor(
            np.concatenate((np.kron(H1r, H0r),
                            np.kron(G1r, G0r),
                            ), axis=0)))
        self.register_buffer('rc_oo', torch.Tensor(
            np.concatenate((np.kron(H1r, H1r),
                            np.kron(G1r, G1r),
                            ), axis=0)))

    def forward(self, x):

        B, Nx, Ny, T, c, ich = x.shape  # (B, Nx, Ny, T, c, k**2)
        ns = math.floor(np.log2(Nx))

        Ud = torch.jit.annotate(List[Tensor], [])
        Us = torch.jit.annotate(List[Tensor], [])

        #         decompose
        for i in range(ns - self.L):
            d, x = self.wavelet_transform(x)
            Ud += [self.A(d) + self.B(x)]
            Us += [self.C(d)]
        x = self.T0(x.view(B, 2 ** self.L, 2 ** self.L, T, -1)).view(
            B, 2 ** self.L, 2 ** self.L, T, c, ich)  # coarsest scale transform

        #        reconstruct
        for i in range(ns - 1 - self.L, -1, -1):
            x = x + Us[i]
            x = torch.cat((x, Ud[i]), -1)
            x = self.evenOdd(x)

        return x

    def wavelet_transform(self, x):
        xa = torch.cat([x[:, ::2, ::2, :, :, :],
                        x[:, ::2, 1::2, :, :, :],
                        x[:, 1::2, ::2, :, :, :],
                        x[:, 1::2, 1::2, :, :, :]
                        ], -1)
        d = torch.matmul(xa, self.ec_d)
        s = torch.matmul(xa, self.ec_s)
        return d, s

    def evenOdd(self, x):

        B, Nx, Ny, T, c, ich = x.shape  # (B, Nx, Ny, c, k**2)
        assert ich == 2 * self.k ** 2
        x_ee = torch.matmul(x, self.rc_ee)
        x_eo = torch.matmul(x, self.rc_eo)
        x_oe = torch.matmul(x, self.rc_oe)
        x_oo = torch.matmul(x, self.rc_oo)

        x = torch.zeros(B, Nx * 2, Ny * 2, T, c, self.k ** 2,
                        device=x.device)
        x[:, ::2, ::2, :, :, :] = x_ee
        x[:, ::2, 1::2, :, :, :] = x_eo
        x[:, 1::2, ::2, :, :, :] = x_oe
        x[:, 1::2, 1::2, :, :, :] = x_oo
        return x

    def reset_parameters(self, initializer):
        initializer(self.T0.weight)
