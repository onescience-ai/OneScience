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
    class Cfg:
        pass

    config = Cfg()
    config.img_size = (2041, 4320)
    config.patch_size = (6, 12)
    config.window_size = (6, 12)
    config.embed_dim = 192
    config.num_heads = (6, 12, 12, 6)
    config.in_chans = 96
    config.depth = 2
    config.out_chans = 94
    config.num_groups = 64
    config.mask='/root/private_data/hanym/modules/onescience/src/onescience/models/xihe/20210628_zos_ocean_mask.npy'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Xihe(config).to()
   
    # 随机输入
    B, Lat, Lon = 1, config.img_size[0], config.img_size[1]
    x = torch.randn(B, config.in_chans, Lat, Lon)
    y = model(x)

    # 期望输出形状： (B, ceil(Lat/ph) * ceil(Lon/pw), embed_dim)
    H_out = math.ceil(config.img_size[0] / config.patch_size[0])
    W_out = math.ceil(config.img_size[1] / config.patch_size[1])
    print("y.shape:", tuple(y.shape))
    print("expected:",(1, 96, 2041, 4320))

if __name__ == "__main__":
    current_path = os.getcwd()
    print("current_path:",current_path)
    sys.path.append(current_path)
    main()


