# import math
# from dataclasses import dataclass

# import numpy as np
# import torch

# from onescience.models.meta import ModelMetaData
# from onescience.modules.module import Module

# from onescience.modules import (
#     PatchRecovery2D,
#     PatchRecovery3D,
#     FuserLayer,
#     DownSample3D,
#     UpSample3D,
#     PatchEmbed2D,
#     PatchEmbed3D,
# )

# @dataclass
# class MetaData(ModelMetaData):
#     name: str = "Pangu"
#     # Optimization
#     jit: bool = False  # ONNX Ops Conflict
#     cuda_graphs: bool = True
#     amp: bool = True
#     # Inference
#     onnx_cpu: bool = False  # No FFT op on CPU
#     onnx_gpu: bool = True
#     onnx_runtime: bool = True
#     # Physics informed
#     var_dim: int = 1
#     func_torch: bool = False
#     auto_grad: bool = False


# class Pangu(Module):
#     """
#     Pangu A PyTorch impl of: `Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Global Weather Forecast`
#     - https://arxiv.org/abs/2211.02556

#     Args:
#         img_size (tuple[int]): Image size [Lat, Lon].
#         patch_size (tuple[int]): Patch token size [Lat, Lon].
#         embed_dim (int): Patch embedding dimension. Default: 192
#         num_heads (tuple[int]): Number of attention heads in different layers.
#         window_size (tuple[int]): Window size.
#     """

#     def __init__(
#         self,
#         img_size=(721, 1440),
#         patch_size=(2, 4, 4),
#         embed_dim=192,
#         num_heads=(6, 12, 12, 6),
#         window_size=(2, 6, 12),
#     ):
#         super().__init__(meta=MetaData())
#         drop_path = np.linspace(0, 0.2, 8).tolist()
#         # In addition, three constant masks(the topography mask, land-sea mask and soil type mask)
#         self.patchembed2d = PatchEmbed2D(
#             img_size=img_size,
#             patch_size=patch_size[1:],
#             in_chans=4 + 3,  # add
#             embed_dim=embed_dim,
#         )
#         self.patchembed3d = OnePatchEmbed(
#             img_size=(13, img_size[0], img_size[1]),
#             patch_size=patch_size,
#             in_chans=5,
#             embed_dim=embed_dim,
#             style='pangu'
#         )
#         patched_inp_shape = (
#             8,
#             math.ceil(img_size[0] / patch_size[1]),
#             math.ceil(img_size[1] / patch_size[2]),
#         )

#         self.layer1 = FuserLayer(
#             dim=embed_dim,
#             input_resolution=patched_inp_shape,
#             depth=2,
#             num_heads=num_heads[0],
#             window_size=window_size,
#             drop_path=drop_path[:2],
#         )

#         patched_inp_shape_downsample = (
#             8,
#             math.ceil(patched_inp_shape[1] / 2),
#             math.ceil(patched_inp_shape[2] / 2),
#         )
        
#         self.downsample = DownSample3D(
#             in_dim=embed_dim,
#             input_resolution=patched_inp_shape,
#             output_resolution=patched_inp_shape_downsample,
#         )
#         self.layer2 = FuserLayer(
#             dim=embed_dim * 2,
#             input_resolution=patched_inp_shape_downsample,
#             depth=6,
#             num_heads=num_heads[1],
#             window_size=window_size,
#             drop_path=drop_path[2:],
#         )
#         self.layer3 = FuserLayer(
#             dim=embed_dim * 2,
#             input_resolution=patched_inp_shape_downsample,
#             depth=6,
#             num_heads=num_heads[2],
#             window_size=window_size,
#             drop_path=drop_path[2:],
#         )
#         self.upsample = UpSample3D(
#             embed_dim * 2, embed_dim, patched_inp_shape_downsample, patched_inp_shape
#         )

#         self.upsample = OneReSample(
#             embed_dim * 2, embed_dim, patched_inp_shape_downsample, patched_inp_shape, style='pangu'
#         )

#         self.layer4 = FuserLayer(
#             dim=embed_dim,
#             input_resolution=patched_inp_shape,
#             depth=2,
#             num_heads=num_heads[3],
#             window_size=window_size,
#             drop_path=drop_path[:2],
#         )
#         # The outputs of the 2nd encoder layer and the 7th decoder layer are concatenated along the channel dimension.
#         self.patchrecovery2d = PatchRecovery2D(
#             img_size, patch_size[1:], 2 * embed_dim, 4
#         )
#         self.patchrecovery3d = PatchRecovery3D(
#             (13, img_size[0], img_size[1]), patch_size, 2 * embed_dim, 5
#         )

#     def forward(self, x):
#         surface = x[:, :7, :, :]
#         upper_air = x[:, 7:, :, :].reshape(x.shape[0], 5, 13, x.shape[2], x.shape[3])
#         surface = self.patchembed2d(surface)
#         upper_air = self.patchembed3d(upper_air)

#         x = torch.concat([surface.unsqueeze(2), upper_air], dim=2)
#         B, C, Pl, Lat, Lon = x.shape
#         x = x.reshape(B, C, -1).transpose(1, 2)

#         x = self.layer1(x)

#         skip = x

#         x = self.downsample(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.upsample(x)
#         x = self.layer4(x)

#         output = torch.concat([x, skip], dim=-1)
#         output = output.transpose(1, 2).reshape(B, -1, Pl, Lat, Lon)
#         output_surface = output[:, :, 0, :, :]
#         output_upper_air = output[:, :, 1:, :, :]

#         output_surface = self.patchrecovery2d(output_surface)
#         output_upper_air = self.patchrecovery3d(output_upper_air)
#         return output_surface, output_upper_air


import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from onescience.models.meta import ModelMetaData
from onescience.modules.module import Module

from onescience.modules import OnePatchEmbed
from onescience.modules import OnePatchRecovery
from onescience.modules import OneReSample
from onescience.modules import OneTransformer3DBlock


class Pangu(Module):
    """
    Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Global Weather Forecast
    
    PyTorch implementation of Pangu-Weather using unified OneXXX module interfaces.
    Reference: https://arxiv.org/abs/2211.02556
    
    Args:
        img_size (tuple[int]): Image size [Lat, Lon]. Default: (721, 1440)
        patch_size (tuple[int]): Patch token size [Pl, Lat, Lon]. Default: (2, 4, 4)
        embed_dim (int): Patch embedding dimension. Default: 192
        num_heads (tuple[int]): Number of attention heads in different layers. Default: (6, 12, 12, 6)
        window_size (tuple[int]): Window size [Wpl, Wlat, Wlon]. Default: (2, 6, 12)
    
    形状:
        - 输入: (B, 69, H, W)
          - 前7通道: 地表变量 (4个物理量 + 3个常量mask)
          - 后62通道: 大气变量 (5个物理量 × 13个压力层 + 2个常量)
        - 输出: 
          - output_surface: (B, 4, H, W) - 地表预测
          - output_upper_air: (B, 5, 13, H, W) - 大气预测
    
    Examples:
        >>> model = Pangu(
        ...     img_size=(721, 1440),
        ...     patch_size=(2, 4, 4),
        ...     embed_dim=192,
        ...     num_heads=(6, 12, 12, 6),
        ...     window_size=(2, 6, 12)
        ... )
        >>> x = torch.randn(2, 69, 721, 1440)
        >>> surface_out, upper_air_out = model(x)
        >>> surface_out.shape, upper_air_out.shape
        (torch.Size([2, 4, 721, 1440]), torch.Size([2, 5, 13, 721, 1440]))
    """
    
    def __init__(
        self,
        img_size=(721, 1440),
        patch_size=(2, 4, 4),
        embed_dim=192,
        num_heads=(6, 12, 12, 6),
        window_size=(2, 6, 12),
    ):
        super().__init__()
        
        drop_path = np.linspace(0, 0.2, 8).tolist()
        
        # Patch Embedding层
        self.patchembed2d = OnePatchEmbed(style='pangu', img_size=(img_size[0], img_size[1]), patch_size=patch_size[1:], in_chans=7, embed_dim=embed_dim)
        self.patchembed3d = OnePatchEmbed(style='pangu', img_size=(13, img_size[0], img_size[1]), patch_size=patch_size, in_chans=5, embed_dim=embed_dim)
        
        # 计算patch后的分辨率
        patched_inp_shape = (8, math.ceil(img_size[0] / patch_size[1]), math.ceil(img_size[1] / patch_size[2]))
        
        # Layer 1: 2个Transformer Block
        self.layer1 = nn.ModuleList([
            OneTransformer3DBlock(
                style='pangu', 
                dim=embed_dim, 
                input_resolution=patched_inp_shape, 
                num_heads=num_heads[0], 
                drop_path=drop_path[:2][i]
                )
            for i in range(2)
        ])

        patched_inp_shape_downsample = (8, math.ceil(patched_inp_shape[1] / 2), math.ceil(patched_inp_shape[2] / 2))
        # DownSample
        self.downsample = OneReSample(style='pangu', in_dim=embed_dim, out_dim=embed_dim * 2, input_resolution=patched_inp_shape, output_resolution=patched_inp_shape_downsample)
        
        # Layer 2: 6个Transformer Block
        self.layer2 = nn.ModuleList([
            OneTransformer3DBlock(
                style='pangu', 
                dim=embed_dim * 2,
                input_resolution=patched_inp_shape_downsample,
                num_heads=num_heads[1],
                drop_path=drop_path[2:][i],
            )
            for i in range(6)
        ])
        
        # Layer 3: 6个Transformer Block
        self.layer3 = nn.ModuleList([
            OneTransformer3DBlock(
                style='pangu', 
                dim=embed_dim * 2,
                input_resolution=patched_inp_shape_downsample,
                num_heads=num_heads[2],
                drop_path=drop_path[2:][i],
            )
            for i in range(6)
        ])
        
        # UpSample
        self.upsample = OneReSample(style='pangu', in_dim=embed_dim * 2, out_dim=embed_dim, input_resolution=patched_inp_shape_downsample, output_resolution=patched_inp_shape,  )
        
        # Layer 4: 2个Transformer Block
        self.layer4 = nn.ModuleList([
            OneTransformer3DBlock(
                style='pangu',
                dim=embed_dim,
                input_resolution=patched_inp_shape,
                num_heads=num_heads[3],
                drop_path=drop_path[:2][i]
            )
            for i in range(2)
        ])
        

        # Patch Recovery层
        self.patchrecovery2d = OnePatchRecovery(style='pangu', img_size=(img_size[0], img_size[1]), patch_size=patch_size[1:], in_chans=2 * embed_dim, out_chans=4)
        self.patchrecovery3d = OnePatchRecovery(style='pangu', img_size=(13, img_size[0], img_size[1]), patch_size=patch_size, in_chans=2 * embed_dim, out_chans=5)
    
    def forward(self, x):
        surface = x[:, :7, :, :]
        upper_air = x[:, 7:, :, :].reshape(x.shape[0], 5, 13, x.shape[2], x.shape[3])
        
        surface = self.patchembed2d(surface)
        upper_air = self.patchembed3d(upper_air)
        
        x = torch.concat([surface.unsqueeze(2), upper_air], dim=2)
        B, C, Pl, Lat, Lon = x.shape
        x = x.reshape(B, C, -1).transpose(1, 2)
        
        # Layer 1
        for blk in self.layer1:
            x = blk(x)
        
        skip = x
        
        # DownSample
        x = self.downsample(x)
        
        # Layer 2
        for blk in self.layer2:
            x = blk(x)
        
        # Layer 3
        for blk in self.layer3:
            x = blk(x)
        
        # UpSample
        x = self.upsample(x)
        
        # Layer 4
        for blk in self.layer4:
            x = blk(x)
        
        # Skip Connection
        output = torch.concat([x, skip], dim=-1)
        output = output.transpose(1, 2).reshape(B, -1, Pl, Lat, Lon)
        
        output_surface = output[:, :, 0, :, :]
        output_upper_air = output[:, :, 1:, :, :]
        
        # Patch Recovery
        output_surface = self.patchrecovery2d(output_surface)
        output_upper_air = self.patchrecovery3d(output_upper_air)
        
        return output_surface, output_upper_air