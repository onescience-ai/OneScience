import math
import os
import torch
import sys
import numpy as np
import torch.nn as nn
from dataclasses import dataclass
from onescience.utils.YParams import YParams
from onescience.models.meta import ModelMetaData
from onescience.modules.embedding.xiheembedding import XiheEmbedding
from onescience.modules.fuser.xihefuse import XiheFuser
from onescience.modules.recovery.xihepatchrecovery import XihePatchRecovery
from onescience.modules.sample.pangudownsample import PanguDownSample
from onescience.modules.sample.xiheupsample import XiheUpSample



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
    
class TensorWithMask:
    def __init__(self, x, mask):
        self.x = x
        self.mask = mask
        self.y=None

   
class Xihe(nn.Module):
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
        out_chans=94,
        num_groups=32,
        
    ):
        super().__init__()
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


        self.patchembed2d = XiheEmbedding()
        self.patchrecovery2d = XihePatchRecovery()

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

        self.block1=XiheFuser(dim=self.embed_dim,input_resolution=input_resolution,num_local=1)

        self.downsample = PanguDownSample(
                                    in_dim=self.embed_dim,
                                    input_resolution=(H_out, W_out),
                                    output_resolution=(H_out // 2, W_out // 2))
        
        input_resolution = (1, H_out // 2, W_out // 2)
        self.mask_h_w=input_resolution

        self.block2=XiheFuser(dim=2*self.embed_dim,input_resolution=input_resolution,num_local=2)

        self.block3=XiheFuser(dim=2*self.embed_dim,input_resolution=input_resolution,num_local=2)
        self.block4=XiheFuser(dim=2*self.embed_dim,input_resolution=input_resolution,num_local=2)

        self.upsample = XiheUpSample(in_dim=2*self.embed_dim,out_dim=embed_dim,input_resolution=(H_out // 2, W_out // 2),  output_resolution=(H_out, W_out), )
        input_resolution = (1, H_out, W_out)
        self.block5=XiheFuser(dim=self.embed_dim,input_resolution=input_resolution,num_local=1)
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
        
        obj1 = TensorWithMask(x, mask1)
        x=self.block1(obj1)          # (B, N, C) 经过 3D 全局注意力
        x1=x
        x=self.downsample(x)                 # (B, N, C) 经过 2D 下采样
        
        if mask_full is not None:            #  mask2
            _,H_out,W_out = self.mask_h_w
            mask2 = self.change_mask(mask_full, x, h_out=H_out, w_out=W_out)
        else:
            mask2 = None
        obj2 = TensorWithMask(x, mask2)
        x=self.block2(obj2)   
        obj2 = TensorWithMask(x, mask2)
        x=self.block3(obj2)  
        obj2 = TensorWithMask(x, mask2)
        x=self.block4(obj2) 
        x=self.upsample(x) 
        obj1 = TensorWithMask(x, mask1)
        x=self.block5(obj1)
        x_out = torch.cat([x, x1], dim=-1)         # (B, N, 2C)
        x_out = self.skip_proj(x_out)
        # B, N, C = x.shape
        # H_, W_ = 341, 360   # 对应 patch grid 尺寸
        H_ = math.ceil(self.img_size[0] / self.patch_size[0])
        W_ = math.ceil(self.img_size[1] / self.patch_size[1])
        x_out = x_out.transpose(1, 2).reshape(B, C, H_, W_)
        x=self.patchrecovery2d(x_out)
        return x
