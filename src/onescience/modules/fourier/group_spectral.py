import torch
import torch.nn as nn
import torch.fft
from typing import Tuple, Union
from onescience.modules.equivariant.group_conv import GroupEquivariantConv2d, GroupEquivariantConv3d

class GSpectralConv2d(nn.Module):
    """
    群等变 2D 谱卷积层 (Group Equivariant Spectral Conv2d).

    该模块结合了群卷积的几何对称性和 FNO 的全局感受野。
    它首先将输入变换到频域 (FFT)，然后使用由 `GroupEquivariantConv2d` 构造的**群等变权重**与频域特征进行复数乘法，最后变换回空域 (IFFT)。

    Args:
        in_channels (int): 输入通道数（基础通道数，不含群维度）。
        out_channels (int): 输出通道数。
        modes (int or tuple): 傅里叶模式数。
        reflection (bool, optional): 是否包含反射群 (D4)。默认值: False。

    形状:
        输入: (B, C_in * Group_Size, H, W)。
        输出: (B, C_out * Group_Size, H, W)。

    Example:
        >>> # 假设 Group Size = 4 (C4)
        >>> gspec = GSpectralConv2d(in_channels=32, out_channels=32, modes=12, reflection=False)
        >>> # 输入形状: (Batch, 32*4, 64, 64)
        >>> x = torch.randn(2, 128, 64, 64)
        >>> out = gspec(x)
        >>> print(out.shape)
        torch.Size([2, 128, 64, 64])
    """
    def __init__(self, in_channels: int, out_channels: int, modes: Union[int, Tuple[int, int]], reflection: bool = False):
        super(GSpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if isinstance(modes, int):
            self.modes = (modes, modes)
        else:
            self.modes = modes # (Mh, Mw)
            
        # 使用 kernel_size = 2*mode - 1 覆盖频谱
        kernel_size = 2 * max(self.modes) - 1
        
        self.conv = GroupEquivariantConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            reflection=reflection,
            bias=False,
            spectral=True,
            Hermitian=True,
        )

    def get_weight(self):
        self.conv.get_weight()
        self.weights = self.conv.weights.transpose(0, 1)

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.shape[0]
        self.get_weight()

        # FFT
        freq0_y = (torch.fft.fftshift(torch.fft.fftfreq(x.shape[-2])) == 0).nonzero().item()
        x_ft = torch.fft.rfft2(x)
        x_ft = torch.fft.fftshift(x_ft, dim=-2) 

        mh, mw = self.modes

        # Filtering
        x_ft = x_ft[..., (freq0_y - mh + 1) : (freq0_y + mh), : mw]

        # Spectral Conv
        current_weights = self.weights[..., :x_ft.shape[-2], :x_ft.shape[-1]]

        out_ft = torch.zeros(
            batchsize,
            self.weights.shape[1], 
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        
        out_ft[..., (freq0_y - mh + 1) : (freq0_y + mh), : mw] = \
            self.compl_mul2d(x_ft, current_weights)

        # IFFT
        x = torch.fft.irfft2(
            torch.fft.ifftshift(out_ft, dim=-2), 
            s=(x.size(-2), x.size(-1))
        )
        return x


class GSpectralConv3d(nn.Module):
    """
    群等变 3D 谱卷积层 (Group Equivariant Spectral Conv3d).

    `GSpectralConv2d` 的 3D 扩展版本。适用于 3D 体素数据或流体场。
    它在 (D, H, W) 三个维度上进行 FFT，利用 `GConv3d` 生成的 3D 群权重进行频域卷积。
    
    Args:
        in_channels (int): 输入通道数 (基础通道)。
        out_channels (int): 输出通道数。
        modes (int or tuple): 傅里叶模式数 (Md, Mh, Mw)。
        reflection (bool, optional): 是否包含反射群。默认值: False。

    形状:
        输入: (B, C_in * Group_Size, D, H, W)。
        输出: (B, C_out * Group_Size, D, H, W)。
    
    Example:
        >>> # 假设 Group Size = 4
        >>> # in_channels=16 -> input_dim = 16*4 = 64
        >>> gspec3d = GSpectralConv3d(in_channels=16, out_channels=16, modes=(8, 8, 8))
        >>> x = torch.randn(2, 64, 32, 32, 32)
        >>> out = gspec3d(x)
        >>> print(out.shape)
        torch.Size([2, 64, 32, 32, 32])
    """
    def __init__(self, in_channels: int, out_channels: int, modes: Union[int, Tuple[int, int, int]], reflection: bool = False):
        super(GSpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if isinstance(modes, int):
            self.modes = (modes, modes, modes)
        else:
            self.modes = modes # (Md, Mh, Mw)

        # 使用最大 mode 构造 Kernel
        max_mode = max(self.modes)
        kernel_size = 2 * max_mode - 1
        
        self.conv = GConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size, # (K, K, K)
            reflection=reflection,
            bias=False,
        )
        
        # 修正：将权重数据转换为复数以支持频域计算
        with torch.no_grad():
             self.conv.W.data = self.conv.W.data.cfloat()

    def get_weight(self):
        self.conv.get_weight()
        self.weights = self.conv.weights.transpose(0, 1)

    def compl_mul3d(self, input, weights):
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.shape[0]
        self.get_weight()

        # FFT 3D - 【修复】使用 rfftn 替代 rfft3 以兼容旧版 PyTorch
        # dim=(-3, -2, -1) 对应 (D, H, W)
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))
        
        # Shift D and H dimensions (dim -3 and -2)
        x_ft = torch.fft.fftshift(x_ft, dim=(-3, -2))

        # Find zero freqs
        freq0_d = (torch.fft.fftshift(torch.fft.fftfreq(x.shape[-3])) == 0).nonzero().item()
        freq0_h = (torch.fft.fftshift(torch.fft.fftfreq(x.shape[-2])) == 0).nonzero().item()
        
        md, mh, mw = self.modes

        # Filtering
        x_ft = x_ft[
            ..., 
            (freq0_d - md + 1) : (freq0_d + md),
            (freq0_h - mh + 1) : (freq0_h + mh),
            : mw
        ]

        # 动态切片权重以匹配非正方形 modes
        current_weights = self.weights[..., :x_ft.shape[-3], :x_ft.shape[-2], :x_ft.shape[-1]]

        out_ft = torch.zeros(
            batchsize,
            self.weights.shape[1],
            x.size(-3),
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        out_ft[
            ..., 
            (freq0_d - md + 1) : (freq0_d + md),
            (freq0_h - mh + 1) : (freq0_h + mh),
            : mw
        ] = self.compl_mul3d(x_ft, current_weights)

        # IFFT 3D - 【修复】使用 irfftn 替代 irfft3
        x = torch.fft.irfftn(
            torch.fft.ifftshift(out_ft, dim=(-3, -2)),
            s=(x.size(-3), x.size(-2), x.size(-1)),
            dim=(-3, -2, -1)
        )

        return x