import torch
import os
import sys
import numpy as np
import logging
import time
import math

from onescience.models.xihe import Xihe

def main():
    torch.manual_seed(0) #随机种子

    # 小尺寸配置：保证 window 能整除/被 pad 到
    # img_size   = (32, 64)    # [Lat, Lon]
    # patch_size = (4, 8)      # 与 window_size 搭配
    # window_size= (4, 8)      # 2D 窗口 -> Xihe 内部会转成 (1, 4, 8)
    # embed_dim  = 64
    # in_chans   = 96
    # depth      = 2           # 给 DropPath 一个长度>1的示例
    # num_heads  = (4, 4, 4, 4)
    img_size=(2041, 4320)      # [Lat, Lon]
    patch_size=(6, 12)         # 与 window_size 配合
    window_size=(6, 12)       # 2D 窗口 => 3D 要写成 (1, 6, 12)
    embed_dim=192
    num_heads=(6, 12, 12, 6)
    in_chans=96
    depth=2
    out_chans=94
    num_groups=64
    
    
    

    # 假设 1 代表海洋，0 代表陆地,
    mask_full = np.load('20210628_zos_ocean_mask.npy')  # shape: (2041, 4320)

    # 构建模型
    model = Xihe(
        img_size=img_size,
        patch_size=patch_size,
        window_size=window_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        in_chans=in_chans,
        depth=depth,
        mask_full=mask_full,
        out_chans=out_chans,
        num_groups=num_groups,
        
    ).to()

    # 随机输入
    B, Lat, Lon = 10, img_size[0], img_size[1]
    x = torch.randn(B, in_chans, Lat, Lon)
    # np.random.seed(123)               # 固定种子
    # a = np.random.rand(2, 3)          # U(0,1)
    #调用xihe处理输入，输出为y
    # with torch.inference_mode():
    y = model(x)

    # 期望输出形状： (B, ceil(Lat/ph) * ceil(Lon/pw), embed_dim)
    H_out = math.ceil(img_size[0] / patch_size[0])
    W_out = math.ceil(img_size[1] / patch_size[1])
    print("y.shape:", tuple(y.shape))
    print("expected:", (B, H_out * W_out, embed_dim))
    print("mean/std:", float(y.mean()), float(y.std()))

if __name__ == "__main__":
    current_path = os.getcwd()
    print("current_path:",current_path)
    sys.path.append(current_path)
    main()


