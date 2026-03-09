import torch
import math
import torch.nn as nn
from timm.layers import trunc_normal_
from onescience.modules.embedding import timestep_embedding, unified_pos_embedding
import numpy as np
import torch.nn.functional as F


################################################################
# Patchify and Neural Spectral Block 1D
################################################################
class NeuralSpectralBlock1D(nn.Module):
    """
    一维神经谱 Transformer 块 (Neural Spectral Transformer Block 1D)。

    该模块采用了类似 Perceiver 的隐空间注意力 (Latent Attention) 架构。
    它首先通过 Patchify 策略将一维长序列分割为局部补丁 (Patches)，利用编码器注意力 (Encoder Attention) 
    将物理空间的序列特征压缩映射为少量的潜在 Token (Latent Tokens)。在紧凑的隐空间内部，结合正余弦基函数进行
    高效的谱域 (Spectral) 变换与特征提取。最后，通过解码器注意力 (Decoder Attention) 将特征投影并还原回
    一维物理空间。这种架构在捕捉长程依赖的同时，极大地降低了标准 Transformer $O(N^2)$ 的计算复杂度。

    Args:
        width (int): 输入和输出的特征通道数。
        num_basis (int): 隐空间谱处理中使用的基函数（模态）数量。
        patch_size (list[int], optional): 分块大小。列表第一个元素用于一维分块。默认值: [3, 3] (仅使用第一个)。
        num_token (int, optional): 隐空间中潜在 Token 的数量，决定了计算瓶颈的大小。默认值: 4。
        n_heads (int, optional): Transformer 注意力机制的头数。默认值: 8。

    形状:
        输入 x: (B, C, L)，其中 L 必须能被 patch_size[0] 整除。
        输出: (B, C, L)，形状与输入保持一致。

    Example:
        >>> # 定义一个通道数为64，基函数数为10，分块大小为5的1D神经谱Transformer块
        >>> block = NeuralSpectralBlock1D(width=64, num_basis=10, patch_size=[5])
        >>> x = torch.randn(8, 64, 100)
        >>> out = block(x)
        >>> out.shape
        torch.Size([8, 64, 100])
    """
    def __init__(self, width, num_basis, patch_size=[3, 3], num_token=4, n_heads=8):
        super(NeuralSpectralBlock1D, self).__init__()
        self.patch_size = patch_size
        self.width = width
        self.num_basis = num_basis

        # basis
        modes_tensor = (1.0 / float(num_basis)) * torch.tensor(
            [i for i in range(num_basis)], dtype=torch.float
        )
        self.register_buffer("modes_list", modes_tensor)
        self.weights = nn.Parameter(
            (1 / (width)) * torch.rand(width, self.num_basis * 2, dtype=torch.float)
        )
        # latent
        self.head = n_heads
        self.num_token = num_token
        self.latent = nn.Parameter(
            (1 / (width))
            * torch.rand(
                self.head, self.num_token, width // self.head, dtype=torch.float
            )
        )
        self.encoder_attn = nn.Conv1d(
            self.width, self.width * 2, kernel_size=1, stride=1
        )
        self.decoder_attn = nn.Conv1d(self.width, self.width, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=-1)

    def self_attn(self, q, k, v):
        # q,k,v: B H L C/H
        attn = self.softmax(torch.einsum("bhlc,bhsc->bhls", q, k))
        return torch.einsum("bhls,bhsc->bhlc", attn, v)

    def latent_encoder_attn(self, x):
        # x: B C H W
        B, C, L = x.shape
        latent_token = self.latent[None, :, :, :].repeat(B, 1, 1, 1)
        x_tmp = (
            self.encoder_attn(x)
            .view(B, C * 2, -1)
            .permute(0, 2, 1)
            .contiguous()
            .view(B, L, self.head, C // self.head, 2)
            .permute(4, 0, 2, 1, 3)
            .contiguous()
        )
        latent_token = self.self_attn(latent_token, x_tmp[0], x_tmp[1]) + latent_token
        latent_token = (
            latent_token.permute(0, 1, 3, 2).contiguous().view(B, C, self.num_token)
        )
        return latent_token

    def latent_decoder_attn(self, x, latent_token):
        # x: B C L
        x_init = x
        B, C, L = x.shape
        latent_token = (
            latent_token.view(B, self.head, C // self.head, self.num_token)
            .permute(0, 1, 3, 2)
            .contiguous()
        )
        x_tmp = (
            self.decoder_attn(x)
            .view(B, C, -1)
            .permute(0, 2, 1)
            .contiguous()
            .view(B, L, self.head, C // self.head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        x = self.self_attn(x_tmp, latent_token, latent_token)
        x = x.permute(0, 1, 3, 2).contiguous().view(B, C, L) + x_init  # B H L C/H
        return x

    def get_basis(self, x):
        # x: B C N
        x_sin = torch.sin(
            self.modes_list[None, None, None, :] * x[:, :, :, None] * math.pi
        )
        x_cos = torch.cos(
            self.modes_list[None, None, None, :] * x[:, :, :, None] * math.pi
        )
        return torch.cat([x_sin, x_cos], dim=-1)

    def compl_mul2d(self, input, weights):
        return torch.einsum("bilm,im->bil", input, weights)

    def forward(self, x):
        B, C, L = x.shape
        # patchify
        x = (
            x.view(
                x.shape[0],
                x.shape[1],
                x.shape[2] // self.patch_size[0],
                self.patch_size[0],
            )
            .contiguous()
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(
                x.shape[0] * (x.shape[2] // self.patch_size[0]),
                x.shape[1],
                self.patch_size[0],
            )
        )
        # Neural Spectral
        # (1) encoder
        latent_token = self.latent_encoder_attn(x)
        # (2) transition
        latent_token_modes = self.get_basis(latent_token)
        latent_token = self.compl_mul2d(latent_token_modes, self.weights) + latent_token
        # (3) decoder
        x = self.latent_decoder_attn(x, latent_token)
        # de-patchify
        x = (
            x.view(B, (L // self.patch_size[0]), C, self.patch_size[0])
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(B, C, L)
            .contiguous()
        )
        return x


################################################################
# Patchify and Neural Spectral Block 2D
################################################################
class NeuralSpectralBlock2D(nn.Module):
    """ 
    二维神经谱 Transformer 块 (Neural Spectral Transformer Block 2D)。

    适用于二维图像或网格数据的隐式 Transformer 核心模块。该模块完美融合了 Vision Transformer (ViT) 
    的分块思想与神经算子的全局感受野。它将输入的二维特征图在空间上进行二维分块 (2D Patchify)，作为 
    Transformer 的输入。通过非对称的交叉注意力机制，将海量的网格特征聚合并投影到低维的潜在 Token 空间中。
    在潜空间内利用谱方法处理全局频率信息，最后再通过注意力机制重构回完整的二维物理分辨率。

    Args:
        width (int): 输入和输出的特征通道数。
        num_basis (int): 隐空间谱处理中使用的基函数数量。
        patch_size (list[int], optional): 二维分块大小 [H_patch, W_patch]。默认值: [3, 3]。
        num_token (int, optional): 隐空间中潜在 Token 的数量，作为信息流转的瓶颈层。默认值: 4。
        n_heads (int, optional): Transformer 注意力机制的头数。默认值: 8。

    形状:
        输入 x: (B, C, H, W)，其中 H, W 必须分别能被 patch_size 对应的维度整除。
        输出: (B, C, H, W)，形状与输入保持一致。

    Example:
        >>> # 处理 64x64 的输入，分块大小为 4x4
        >>> block = NeuralSpectralBlock2D(width=32, num_basis=12, patch_size=[4, 4])
        >>> x = torch.randn(4, 32, 64, 64)
        >>> out = block(x)
        >>> out.shape
        torch.Size([4, 32, 64, 64])
    """
    def __init__(self, width, num_basis, patch_size=[3, 3], num_token=4, n_heads=8):
        super(NeuralSpectralBlock2D, self).__init__()
        self.patch_size = patch_size
        self.width = width
        self.num_basis = num_basis

        # basis
        modes_tensor = (1.0 / float(num_basis)) * torch.tensor(
            [i for i in range(num_basis)], dtype=torch.float
        )
        self.register_buffer("modes_list", modes_tensor)
        self.weights = nn.Parameter(
            (1 / (width)) * torch.rand(width, self.num_basis * 2, dtype=torch.float)
        )
        # latent
        self.head = n_heads
        self.num_token = num_token
        self.latent = nn.Parameter(
            (1 / (width))
            * torch.rand(
                self.head, self.num_token, width // self.head, dtype=torch.float
            )
        )
        self.encoder_attn = nn.Conv2d(
            self.width, self.width * 2, kernel_size=1, stride=1
        )
        self.decoder_attn = nn.Conv2d(self.width, self.width, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=-1)

    def self_attn(self, q, k, v):
        # q,k,v: B H L C/H
        attn = self.softmax(torch.einsum("bhlc,bhsc->bhls", q, k))
        return torch.einsum("bhls,bhsc->bhlc", attn, v)

    def latent_encoder_attn(self, x):
        # x: B C H W
        B, C, H, W = x.shape
        L = H * W
        latent_token = self.latent[None, :, :, :].repeat(B, 1, 1, 1)
        x_tmp = (
            self.encoder_attn(x)
            .view(B, C * 2, -1)
            .permute(0, 2, 1)
            .contiguous()
            .view(B, L, self.head, C // self.head, 2)
            .permute(4, 0, 2, 1, 3)
            .contiguous()
        )
        latent_token = self.self_attn(latent_token, x_tmp[0], x_tmp[1]) + latent_token
        latent_token = (
            latent_token.permute(0, 1, 3, 2).contiguous().view(B, C, self.num_token)
        )
        return latent_token

    def latent_decoder_attn(self, x, latent_token):
        # x: B C L
        x_init = x
        B, C, H, W = x.shape
        L = H * W
        latent_token = (
            latent_token.view(B, self.head, C // self.head, self.num_token)
            .permute(0, 1, 3, 2)
            .contiguous()
        )
        x_tmp = (
            self.decoder_attn(x)
            .view(B, C, -1)
            .permute(0, 2, 1)
            .contiguous()
            .view(B, L, self.head, C // self.head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        x = self.self_attn(x_tmp, latent_token, latent_token)
        x = x.permute(0, 1, 3, 2).contiguous().view(B, C, H, W) + x_init  # B H L C/H
        return x

    def get_basis(self, x):
        # x: B C N
        x_sin = torch.sin(
            self.modes_list[None, None, None, :] * x[:, :, :, None] * math.pi
        )
        x_cos = torch.cos(
            self.modes_list[None, None, None, :] * x[:, :, :, None] * math.pi
        )
        return torch.cat([x_sin, x_cos], dim=-1)

    def compl_mul2d(self, input, weights):
        return torch.einsum("bilm,im->bil", input, weights)

    def forward(self, x):
        B, C, H, W = x.shape
        # patchify
        x = (
            x.view(
                x.shape[0],
                x.shape[1],
                x.shape[2] // self.patch_size[0],
                self.patch_size[0],
                x.shape[3] // self.patch_size[1],
                self.patch_size[1],
            )
            .contiguous()
            .permute(0, 2, 4, 1, 3, 5)
            .contiguous()
            .view(
                x.shape[0]
                * (x.shape[2] // self.patch_size[0])
                * (x.shape[3] // self.patch_size[1]),
                x.shape[1],
                self.patch_size[0],
                self.patch_size[1],
            )
        )
        # Neural Spectral
        # (1) encoder
        latent_token = self.latent_encoder_attn(x)
        # (2) transition
        latent_token_modes = self.get_basis(latent_token)
        latent_token = self.compl_mul2d(latent_token_modes, self.weights) + latent_token
        # (3) decoder
        x = self.latent_decoder_attn(x, latent_token)
        # de-patchify
        x = (
            x.view(
                B,
                (H // self.patch_size[0]),
                (W // self.patch_size[1]),
                C,
                self.patch_size[0],
                self.patch_size[1],
            )
            .permute(0, 3, 1, 4, 2, 5)
            .contiguous()
            .view(B, C, H, W)
            .contiguous()
        )
        return x


################################################################
# Patchify and Neural Spectral Block 3D
################################################################
class NeuralSpectralBlock3D(nn.Module):
    """
    三维神经谱 Transformer 块 (Neural Spectral Transformer Block 3D)。

    针对三维体数据（如流体、气象、视频等时空数据）设计的隐式 Transformer 算子。
    该模块在三个维度上对输入进行立体分块 (3D Patchify)，利用注意力机制将极其庞大的三维体素/时空序列
    信息压缩提取到极紧凑的潜空间 (Latent Space) 中。在潜空间执行基于正交基函数的谱域信息交互，再通过
    解码器注意力还原至三维网格。通过这种 Latent Transformer 架构，成功绕过了三维自注意力机制带来的
    内存爆炸灾难，实现了对 3D 时空数据高效的全局信息路由和长程依赖建模。

    Args:
        width (int): 输入和输出的特征通道数。
        num_basis (int): 隐空间谱处理中使用的基函数数量。
        patch_size (list[int], optional): 三维分块大小 [H_patch, W_patch, T_patch]。默认值: [8, 8, 4]。
        num_token (int, optional): 隐空间中潜在 Token 的数量，作为三维特征的高度浓缩表示。默认值: 4。
        n_heads (int, optional): Transformer 注意力机制的头数。默认值: 8。

    形状:
        输入 x: (B, C, D1, D2, D3)，通常对应 (B, C, H, W, T) 或 (B, C, X, Y, Z)。三个空间维度必须分别能被 patch_size 整除。
        输出: (B, C, D1, D2, D3)，形状与输入保持一致。

    Example:
        >>> # 处理 32x32x16 的 3D 时空数据
        >>> block = NeuralSpectralBlock3D(width=16, num_basis=8, patch_size=[4, 4, 2])
        >>> x = torch.randn(2, 16, 32, 32, 16)
        >>> out = block(x)
        >>> out.shape
        torch.Size([2, 16, 32, 32, 16])
    """
    def __init__(self, width, num_basis, patch_size=[8, 8, 4], num_token=4, n_heads=8):
        super(NeuralSpectralBlock3D, self).__init__()
        self.patch_size = patch_size
        self.width = width
        self.num_basis = num_basis

        # basis
        modes_tensor = (1.0 / float(num_basis)) * torch.tensor(
            [i for i in range(num_basis)], dtype=torch.float
        )
        self.register_buffer("modes_list", modes_tensor)
        self.weights = nn.Parameter(
            (1 / (width)) * torch.rand(width, self.num_basis * 2, dtype=torch.float)
        )
        # latent
        self.head = n_heads
        self.num_token = num_token
        self.latent = nn.Parameter(
            (1 / (width))
            * torch.rand(
                self.head, self.num_token, width // self.head, dtype=torch.float
            )
        )
        self.encoder_attn = nn.Conv3d(
            self.width, self.width * 2, kernel_size=1, stride=1
        )
        self.decoder_attn = nn.Conv3d(self.width, self.width, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=-1)

    def self_attn(self, q, k, v):
        # q,k,v: B H L C/H
        attn = self.softmax(torch.einsum("bhlc,bhsc->bhls", q, k))
        return torch.einsum("bhls,bhsc->bhlc", attn, v)

    def latent_encoder_attn(self, x):
        # x: B C H W
        B, C, H, W, T = x.shape
        L = H * W * T
        latent_token = self.latent[None, :, :, :].repeat(B, 1, 1, 1)
        x_tmp = (
            self.encoder_attn(x)
            .view(B, C * 2, -1)
            .permute(0, 2, 1)
            .contiguous()
            .view(B, L, self.head, C // self.head, 2)
            .permute(4, 0, 2, 1, 3)
            .contiguous()
        )
        latent_token = self.self_attn(latent_token, x_tmp[0], x_tmp[1]) + latent_token
        latent_token = (
            latent_token.permute(0, 1, 3, 2).contiguous().view(B, C, self.num_token)
        )
        return latent_token

    def latent_decoder_attn(self, x, latent_token):
        # x: B C L
        x_init = x
        B, C, H, W, T = x.shape
        L = H * W * T
        latent_token = (
            latent_token.view(B, self.head, C // self.head, self.num_token)
            .permute(0, 1, 3, 2)
            .contiguous()
        )
        x_tmp = (
            self.decoder_attn(x)
            .view(B, C, -1)
            .permute(0, 2, 1)
            .contiguous()
            .view(B, L, self.head, C // self.head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        x = self.self_attn(x_tmp, latent_token, latent_token)
        x = x.permute(0, 1, 3, 2).contiguous().view(B, C, H, W, T) + x_init  # B H L C/H
        return x

    def get_basis(self, x):
        # x: B C N
        x_sin = torch.sin(
            self.modes_list[None, None, None, :] * x[:, :, :, None] * math.pi
        )
        x_cos = torch.cos(
            self.modes_list[None, None, None, :] * x[:, :, :, None] * math.pi
        )
        return torch.cat([x_sin, x_cos], dim=-1)

    def compl_mul2d(self, input, weights):
        return torch.einsum("bilm,im->bil", input, weights)

    def forward(self, x):
        B, C, H, W, T = x.shape
        # patchify
        x = (
            x.view(
                x.shape[0],
                x.shape[1],
                x.shape[2] // self.patch_size[0],
                self.patch_size[0],
                x.shape[3] // self.patch_size[1],
                self.patch_size[1],
                x.shape[4] // self.patch_size[2],
                self.patch_size[2],
            )
            .contiguous()
            .permute(0, 2, 4, 6, 1, 3, 5, 7)
            .contiguous()
            .view(
                x.shape[0]
                * (x.shape[2] // self.patch_size[0])
                * (x.shape[3] // self.patch_size[1])
                * (x.shape[4] // self.patch_size[2]),
                x.shape[1],
                self.patch_size[0],
                self.patch_size[1],
                self.patch_size[2],
            )
        )
        # Neural Spectral
        # (1) encoder
        latent_token = self.latent_encoder_attn(x)
        # (2) transition
        latent_token_modes = self.get_basis(latent_token)
        latent_token = self.compl_mul2d(latent_token_modes, self.weights) + latent_token
        # (3) decoder
        x = self.latent_decoder_attn(x, latent_token)
        # de-patchify
        x = (
            x.view(
                B,
                (H // self.patch_size[0]),
                (W // self.patch_size[1]),
                (T // self.patch_size[2]),
                C,
                self.patch_size[0],
                self.patch_size[1],
                self.patch_size[2],
            )
            .permute(0, 4, 1, 5, 2, 6, 3, 7)
            .contiguous()
            .view(B, C, H, W, T)
            .contiguous()
        )
        return x
