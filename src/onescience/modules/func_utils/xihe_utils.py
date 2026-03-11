import math
import os
import torch
import sys
import numpy as np
import torch.nn as nn

def change_mask(mask_full, x, h_out, w_out):
    
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