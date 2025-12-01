import math
import os
import torch
import sys
import numpy as np
from dataclasses import dataclass
from onescience.models.meta import ModelMetaData
from onescience.models.module import Module


from onescience.models.utils import (
    PatchEmbed2D,
    PatchEmbed3D,
    PatchRecovery2D,
    PatchRecovery3D,
)
# from onescience.models.layers  import LocalSIE
# from torch import nn

import torch.nn as nn
from onescience.models.xihe.layers import LocalSIE # 这种可以

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
        img_size=(2161, 4320),      # [Lat, Lon]
        patch_size=(6, 12),         # 与 window_size 配合
        window_size=(6, 12),        # 2D 窗口 => 3D 要写成 (1, 6, 12)
        embed_dim=192,
        num_heads=(6, 12, 12, 6),
        in_chans=96,
        depth=1,
    ):
        super().__init__(meta=MetaData())
        self.img_size = img_size
        self.patch_size = patch_size

        # 2D patch embedding
        self.patchembed2d = PatchEmbed2D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        # patch 后的 3D 分辨率: (Pl=1, Lat_out, Lon_out)
        H_out = math.ceil(img_size[0] / patch_size[0])
        W_out = math.ceil(img_size[1] / patch_size[1])
        input_resolution = (1, H_out, W_out)

        # 3D 窗口：把 2D 窗口扩成 (1, win_lat, win_lon)
        window_size_3d = (1, window_size[0], window_size[1])

        # drop_path schedule 与 depth 对齐
        if depth > 1:
            drop_path = np.linspace(0, 0.2, depth).tolist()
        else:
            drop_path = 0.0

        self.local1 = LocalSIE(
            dim=embed_dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads[0],
            window_size=window_size_3d,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_path=drop_path,
            norm_layer=nn.LayerNorm,
        )

    def forward(self, x: torch.Tensor):
        # x: (B, in_chans, Lat, Lon)
        x = self.patchembed2d(x)                  # (B, C=embed_dim, H', W')
        # print("x.shape:", tuple(x.shape))
        x = x.flatten(2).transpose(1, 2)          # (B, N=H'*W', C)
        # print("x.shape:", tuple(x.shape))
        x = self.local1(x)                        # (B, N, C) 经过 3D 局部注意力
        print("x.shape:", tuple(x.shape))
        return x



# --- 如果你的 Xihe 定义就在当前文件，可以直接粘贴在上面，这里就不用 import ---


def main():
    torch.manual_seed(0)

    # 小尺寸配置：保证 window 能整除/被 pad 到
    img_size   = (32, 64)    # [Lat, Lon]
    patch_size = (4, 8)      # 与 window_size 搭配
    window_size= (3, 8)      # 2D 窗口 -> Xihe 内部会转成 (1, 4, 8)
    embed_dim  = 64
    in_chans   = 96
    depth      = 2           # 给 DropPath 一个长度>1的示例
    num_heads  = (4, 4, 4, 4)

    # 构建模型
    model = Xihe(
        img_size=img_size,
        patch_size=patch_size,
        window_size=window_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        in_chans=in_chans,
        depth=depth,
    ).to()

    # 随机输入
    B, Lat, Lon = 1, img_size[0], img_size[1]
    x = torch.randn(B, in_chans, Lat, Lon)

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








# def smoke_test(device="cpu"):
#     torch.manual_seed(0)

#     # 小尺寸配置：保证 window 能整除/被 pad 到
#     img_size   = (32, 64)    # [Lat, Lon]
#     patch_size = (4, 8)      # 与 window_size 搭配
#     window_size= (4, 8)      # 2D 窗口 -> Xihe 内部会转成 (1, 4, 8)
#     embed_dim  = 64
#     in_chans   = 96
#     depth      = 2           # 给 DropPath 一个长度>1的示例
#     num_heads  = (4, 4, 4, 4)

#     # 构建模型
#     model = Xihe(
#         img_size=img_size,
#         patch_size=patch_size,
#         window_size=window_size,
#         embed_dim=embed_dim,
#         num_heads=num_heads,
#         in_chans=in_chans,
#         depth=depth,
#     ).to(device).eval()

#     # 随机输入
#     B, Lat, Lon = 1, img_size[0], img_size[1]
#     x = torch.randn(B, in_chans, Lat, Lon, device=device)
#     #调用xihe处理输入，输出为y
#     with torch.inference_mode():
#         y = model(x)

#     # 期望输出形状： (B, ceil(Lat/ph) * ceil(Lon/pw), embed_dim)
#     H_out = math.ceil(img_size[0] / patch_size[0])
#     W_out = math.ceil(img_size[1] / patch_size[1])
#     print("y.shape:", tuple(y.shape))
#     print("expected:", (B, H_out * W_out, embed_dim))
#     print("mean/std:", float(y.mean()), float(y.std()))

# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print("using device:", device)
#     smoke_test(device)
