#自己编写的部分第一版本
import math
from dataclasses import dataclass

import numpy as np
import torch

# from ..meta import ModelMetaData
from ..module import Module


from ..utils import (
    PatchEmbed2D,
    PatchEmbed3D,
    PatchRecovery2D,
    PatchRecovery3D,
)
from ..layers import LocalSie
from torch import nn

@dataclass
class MetaData(ModelMetaData):
    name: str = "Xihe"
    # Optimization
    jit: bool = False  # ONNX Ops Conflict
    cuda_graphs: bool = True
    amp: bool = True
    
    # Inference
    onnx_cpu: bool = False  # No FFT op on CPU
    onnx_gpu: bool = True
    onnx_runtime: bool = True
    
    # Physics informed
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False


class Xihe(Module):
    """
    Xihe A PyTorch impl of: `XiHe: A Data-Driven Model for Global Ocean Eddy-Resolving Forecasting`
    - https://arxiv.org/abs/2402.02995

    Args:
        img_size (tuple[int]): Image size [Lat, Lon].
        patch_size (tuple[int]): Patch token size [Lat, Lon].
        embed_dim (int): Patch embedding dimension. Default: 192
        num_heads (tuple[int]): Number of attention heads in different layers.
        window_size (tuple[int]): Window size.
    """  
      
    def __init__(
        self,
        img_size = (2161, 4320),         # 对应 1/12° 全球格点
        patch_size = (6, 12),            # 与 window_size 配合整除
        window_size = (6, 12),           # 窗口注意力范围
        embed_dim = 192,                 # 小模型（ViT）
        num_heads = (6, 12, 12, 6),      
    ):
        super().__init__(meta=MetaData())
        self.patchembed2d = PatchEmbed2D(
            img_size=img_size,
            patch_size=patch_size,   
            in_chans=96,
            embed_dim=embed_dim,
        ) 
        
        drop_path = np.linspace(0, 0.2, 2).tolist()#后续参照pangu
        patched_inp_shape = (
            1,
            math.ceil(img_size[0] / patch_size[1]),
            math.ceil(img_size[1] / patch_size[2]),
        ) 
        
        self.local1= LocalSie(
            dim=embed_dim,
            input_resolution=patched_inp_shape,
            depth=1,
            num_heads=num_heads[0],           # 可设每个stage不同
            window_size=window_size,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_path=drop_path,
            norm_layer=nn.LayerNorm,   
        )   
           
    def forward(self, x):
        x = self.patchembed2d(x)         # (B, embed_dim, H', W') norm
        B, C, Lat, Lon = x.shape
        x = x.reshape(B,C,-1).transpose(1, 2)  #（B,h*w,C）
        return x







# # ---- Patch Partition ----
# class PatchPartition(nn.Module):
#     def __init__(self, in_channels, embed_dim, patch_size=4):
#         super().__init__()
#         self.proj = nn.Conv2d(in_channels, embed_dim, 
#                               kernel_size=patch_size, stride=patch_size)
#         self.norm = nn.LayerNorm(embed_dim)

#     def forward(self, x):
#         # x: [B, C, H, W]
#         x = self.proj(x)   # [B, C', H/patch, W/patch]
#         B, C, H, W = x.shape
#         x = x.flatten(2).transpose(1, 2)  # [B, N, C]
#         x = self.norm(x)
#         return x, (H, W)

# # ---- Local SIE (Window Self-Attention) ----
# class LocalSIE(nn.Module):
#     def __init__(self, dim, num_heads, window_size=7):
#         super().__init__()
#         self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
#         self.norm = nn.LayerNorm(dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, dim*4),
#             nn.GELU(),
#             nn.Linear(dim*4, dim)
#         )

#     def forward(self, x):
#         # 简化：未写窗口切分，论文用 Swin Transformer W-MSA
#         x = x + self.attn(x, x, x)[0]
#         x = x + self.mlp(self.norm(x))
#         return x

# # ---- Global SIE (Group Propagation) ----
# class GlobalSIE(nn.Module):
#     def __init__(self, dim, num_groups=64):
#         super().__init__()
#         self.groups = nn.Parameter(torch.randn(num_groups, dim))
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, dim),
#             nn.GELU(),
#             nn.Linear(dim, dim)
#         )

#     def forward(self, x):
#         # x: [B, N, C]
#         # 特征分组 (cross-attention 简化版)
#         B, N, C = x.shape
#         g = self.groups.unsqueeze(0).expand(B, -1, -1)  # [B, G, C]
#         # 组间传播 (MLP-Mixer 简化)
#         g = self.mlp(g)
#         # 特征解组 (cross-attention 简化)
#         x = x + torch.bmm(torch.softmax(x @ g.transpose(1,2), dim=-1), g)
#         return x

# # ---- Ocean-Specific Block ----
# class OceanBlock(nn.Module):
#     def __init__(self, dim, num_heads):
#         super().__init__()
#         self.local = LocalSIE(dim, num_heads)
#         self.global_sie = GlobalSIE(dim)

#     def forward(self, x):
#         x = self.local(x)
#         x = self.global_sie(x)
#         return x

# # ---- Patch Restoration ----
# class PatchRestoration(nn.Module):
#     def __init__(self, out_channels, embed_dim, patch_size=4):
#         super().__init__()
#         self.deproj = nn.ConvTranspose2d(embed_dim, out_channels,
#                                          kernel_size=patch_size, stride=patch_size)

#     def forward(self, x, size):
#         # x: [B, N, C], size = (H, W)
#         B, N, C = x.shape
#         H, W = size
#         x = x.transpose(1, 2).reshape(B, C, H, W)
#         x = self.deproj(x)
#         return x

# # ---- Xihe Model ----
# class Xihe(nn.Module):
#     def __init__(self, in_channels=96, out_channels=94, embed_dim=128, depth=5, num_heads=4):
#         super().__init__()
#         self.patch_partition = PatchPartition(in_channels, embed_dim)
#         self.blocks = nn.ModuleList([OceanBlock(embed_dim, num_heads) for _ in range(depth)])
#         self.patch_restoration = PatchRestoration(out_channels, embed_dim)

#     def forward(self, x):
#         # x: [B, C, H, W]
#         x, size = self.patch_partition(x)
#         for blk in self.blocks:
#             x = blk(x)
#         x = self.patch_restoration(x, size)
#         return x
