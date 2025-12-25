import math
import os
import torch
import sys
import numpy as np
from dataclasses import dataclass
from onescience.models.meta import ModelMetaData
from onescience.models.module import Module
from onescience.utils.YParams import YParams


from onescience.models.utils import (
    PatchEmbed2D,
    PatchEmbed3D,
    PatchRecovery2D,
    PatchRecovery3D,
)


import torch.nn as nn
from onescience.models.xihe.oceanspecificblock import OceanSpecificBlock
from ..layers.resample_layers import DownSample2D, UpSample



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
    https://arxiv.org/abs/2402.02995
    """
    def __init__(
        self,
        config,
        img_size=(2041, 4320),     
        patch_size=(6, 12),        
        window_size=(6, 12),       
        embed_dim=192,
        num_heads=(6, 12, 12, 6),
        in_chans=96,
        depth=1,
        mask_full=None,
        out_chans=96,
        num_groups=32,
        
    ):
        super().__init__(meta=MetaData())
        self.img_size = config.img_size
        self.patch_size =config.patch_size
        # 正确初始化 mask_full
        self.mask=config.mask
        mask_full = np.load(self.mask)
        self.mask_full = mask_full if mask_full is not None else None
        self.mask_h_w=None
        self.out_chans=config.out_chans
        self.in_chans=config.in_chans
        self.num_groups=config.num_groups
        self.embed_dim=config.embed_dim
        
        self.skip_proj = nn.Linear(2*self.embed_dim, self.embed_dim)

        
        # 2D patch embedding
        self.patchembed2d = PatchEmbed2D(
            img_size=img_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
        )
        self.patchrecovery2d = PatchRecovery2D(
            img_size=img_size,
            patch_size=self.patch_size,
            in_chans=self.embed_dim,
            out_chans=self.out_chans,
        )

        # patch 后的 3D 分辨率: (Pl=1, Lat_out, Lon_out)
        H_out = math.ceil(img_size[0] / patch_size[0])
        W_out = math.ceil(img_size[1] / patch_size[1])
        input_resolution = (1, H_out, W_out)
        
        self.mask_h_w=input_resolution
        # 3D 窗口：把 2D 窗口扩成 (1, win_lat, win_lon)
        window_size_3d = (1, window_size[0], window_size[1]) #  window_size (tuple[int]): Window size [pressure levels, latitude, longitude].

        # 防止过拟合，随机丢弃一部分
        if depth > 1:
            drop_path = np.linspace(0, 0.2, depth).tolist()
        else:
            drop_path = 0.0   
     
        self.block1=OceanSpecificBlock(
            dim=self.embed_dim,
            input_resolution=input_resolution,
            num_local=1,
            num_global=1,
            depth_local=depth,#可以堆叠多个transformer3D
            num_heads_local=num_heads[0],
            num_heads_global=num_heads[1],
            window_size=window_size_3d,
            mlp_ratio=4.0,
            qkv_bias=True,
            num_groups=num_groups,
            drop_path=drop_path,#支持 stochastic depth
            norm_layer=nn.LayerNorm
        )
        
        self.downsample = DownSample2D(
            in_dim=self.embed_dim,
            input_resolution=(H_out, W_out),
            output_resolution=(H_out // 2, W_out // 2),
        )
        
        input_resolution = (1, H_out // 2, W_out // 2)
        self.mask_h_w=input_resolution
        # embed_dim=128
        self.block2=OceanSpecificBlock(
            dim=2*self.embed_dim,
            input_resolution=input_resolution,
            num_local=2,
            num_global=1,
            depth_local=depth,
            num_heads_local=num_heads[0],
            num_heads_global=num_heads[1],
            window_size=window_size_3d,
            mlp_ratio=4.0,
            qkv_bias=True,
            num_groups=num_groups,
            drop_path=drop_path,
            norm_layer=nn.LayerNorm
        )
        
        self.block3=OceanSpecificBlock(
            dim=2*self.embed_dim,
            input_resolution=input_resolution,
            num_local=2,
            num_global=1,
            depth_local=depth,
            num_heads_local=num_heads[0],
            num_heads_global=num_heads[1],
            window_size=window_size_3d,
            mlp_ratio=4.0,
            qkv_bias=True,
            num_groups=num_groups,
            drop_path=drop_path,
            norm_layer=nn.LayerNorm
        )
        
        self.block4=OceanSpecificBlock(
            dim=2*self.embed_dim,
            input_resolution=input_resolution,
            num_local=2,
            num_global=1,
            depth_local=depth,
            num_heads_local=num_heads[0],
            num_heads_global=num_heads[1],
            window_size=window_size_3d,
            mlp_ratio=4.0,
            qkv_bias=True,
            num_groups=num_groups,
            drop_path=drop_path,
            norm_layer=nn.LayerNorm
        )
        self.upsample = UpSample(
                in_dim=2*self.embed_dim,
                out_dim=embed_dim,
                input_resolution=(H_out // 2, W_out // 2),  
                output_resolution=(H_out, W_out),         
            )   
        input_resolution = (1, H_out, W_out)
        self.block5=OceanSpecificBlock(
            dim=self.embed_dim,
            input_resolution=input_resolution,
            num_local=1,
            num_global=1,
            depth_local=depth,#可以堆叠多个transformer3D
            num_heads_local=num_heads[0],
            num_heads_global=num_heads[1],
            window_size=window_size_3d,
            mlp_ratio=4.0,
            qkv_bias=True,
            num_groups=num_groups,
            drop_path=drop_path,#支持 stochastic depth
            norm_layer=nn.LayerNorm
        ) 
   
    def change_mask(self,mask_full, x, h_out, w_out):
        #根据当前层特征分辨率，自动生成掩码（海洋=1，陆地=0）
            if not torch.is_tensor(mask_full):
                mask_full = torch.tensor(mask_full, dtype=torch.float32)
            else:
                mask_full = mask_full

            H, W = mask_full.shape
            patch_h = math.ceil(H / h_out)
            patch_w = math.ceil(W / w_out)

            mask_coarse = torch.zeros((h_out, w_out), dtype=torch.float32)
            for i in range(h_out):
                for j in range(w_out):
                    h0, h1 = i * patch_h, min((i + 1) * patch_h, H)
                    w0, w1 = j * patch_w, min((j + 1) * patch_w, W)
                    patch = mask_full[h0:h1, w0:w1]
                    mask_coarse[i, j] = 1.0 if torch.any(patch > 0.5) else 0.0
            
            mask_coarse = mask_coarse.to(x.device, dtype=x.dtype) 
            B = x.shape[0]                
            mask_coarse = mask_coarse.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1) #broadcast
            return mask_coarse  
        
    def forward(self, x: torch.Tensor):
        x = self.patchembed2d(x)                  # (B, C=embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)          # (B, N=H'*W', C) 
        B, N, C = x.shape     
        mask_full=self.mask_full        
        
       
        if mask_full is not None:              # mask1
            H_out = math.ceil(self.img_size[0] / self.patch_size[0])
            W_out = math.ceil(self.img_size[1] / self.patch_size[1])
            mask1 = self.change_mask(mask_full, x, h_out=H_out, w_out=W_out)
        else:
            mask1 = None
        # ratio = (mask1 == 1).float().mean().item()
        # print("海洋区域占比:", ratio)
        # print("mask1",mask1.shape)
        # print("x.shape-patch------", tuple(x.shape))

        x=self.block1(x,mask=mask1)          # (B, N, C) 经过 3D 全局注意力
        x1=x
        # print("x.shape---249",x.shape)
        x=self.downsample(x)                 # (B, N, C) 经过 2D 下采样
        
        if mask_full is not None:            #  mask2
            _,H_out,W_out = self.mask_h_w
            mask2 = self.change_mask(mask_full, x, h_out=H_out, w_out=W_out)
        else:
            mask2 = None
        x=self.block2(x,mask=mask2)                      
        x=self.block3(x,mask=mask2)                                   
        x=self.block4(x,mask=mask2)  
        # print("x.shape---261",x.shape)
        x=self.upsample(x) 
        # print("x.shape---263",x.shape)
        x=self.block5(x,mask=mask1)
        x_out = torch.cat([x, x1], dim=-1)         # (B, N, 2C)
        x_out = self.skip_proj(x_out)

        # B, N, C = x.shape
        # H_, W_ = 341, 360   # 对应 patch grid 尺寸
        H_ = math.ceil(self.img_size[0] / self.patch_size[0])
        W_ = math.ceil(self.img_size[1] / self.patch_size[1])
        x_out = x_out.transpose(1, 2).reshape(B, C, H_, W_)
        x=self.patchrecovery2d(x_out)
        return x

