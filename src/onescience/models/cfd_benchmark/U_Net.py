import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from onescience.modules import (
    OneMlp,
    OneFourier,
    OneEncoder,
    OneDecoder,
    OneHead
)
from onescience.modules.fourier.geo_spectral import IPHI
from onescience.modules.embedding import timestep_embedding, unified_pos_embedding

class Model(nn.Module):
    """
    多尺度物理场 U-Net 模型。

    该模型支持结构化网格（1D/2D/3D）和非结构化网格（通过 GeoFNO 的几何投影）的物理场预测。
    利用 OneEncoder 和 OneDecoder 实现多尺度特征提取与融合，极大地简化了网络搭建过程。

    Args:
        args: 包含模型配置的参数命名空间 (如 task, geotype, n_hidden 等)。
        device: 运行设备。
        bilinear (bool, optional): U-Net 上采样是否使用双线性插值。默认值: True。
        s1 (int, optional): 非结构化网格投影的潜在空间高度。默认值: 96。
        s2 (int, optional): 非结构化网格投影的潜在空间宽度。默认值: 96。

    形状:
        输入 x: 坐标张量 (B, N, space_dim)。
        输入 fx: 物理场特征张量 (B, N, fun_dim)。
        输入 T: 可选的时间步张量。
        输出: (B, N, out_dim)
    """
    def __init__(self, args, device, bilinear=True, s1=96, s2=96):
        super(Model, self).__init__()
        self.__name__ = "U-Net"
        self.args = args
        self.bilinear = bilinear
        
        normtype = "bn" if args.task == "steady" else "in"

        # ==========================================
        # 1. 位置编码与预处理 (Embedding)
        # ==========================================
        if args.unified_pos and args.geotype != "unstructured":  # structured mesh
            self.pos = unified_pos_embedding(args.shapelist, args.ref, device=device)
            input_dim = args.fun_dim + args.ref ** len(args.shapelist)
        else:
            input_dim = args.fun_dim + args.space_dim

        self.preprocess = OneMlp(
            style="StandardMLP",
            input_dim=input_dim,
            hidden_dims=[args.n_hidden * 2],
            output_dim=args.n_hidden,
            activation=args.act,
        )

        if args.time_input:
            self.time_fc = nn.Sequential(
                nn.Linear(args.n_hidden, args.n_hidden),
                nn.SiLU(),
                nn.Linear(args.n_hidden, args.n_hidden),
            )

        # ==========================================
        # 2. 几何投影 (Geometry Projection)
        # ==========================================
        if self.args.geotype == "unstructured":
            self.fftproject_in = OneFourier(
                style="GeoSpectralConv2d",
                in_channels=args.n_hidden, 
                out_channels=args.n_hidden, 
                modes1=args.modes, 
                modes2=args.modes, 
                s1=s1, 
                s2=s2
            )
            self.fftproject_out = OneFourier(
                style="GeoSpectralConv2d",
                in_channels=args.n_hidden, 
                out_channels=args.n_hidden, 
                modes1=args.modes, 
                modes2=args.modes, 
                s1=s1, 
                s2=s2
            )
            self.iphi = IPHI()
            
            patch_size = [(size + (16 - size % 16) % 16) // 16 for size in [s1, s2]]
            self.padding = [(16 - size % 16) % 16 for size in [s1, s2]]
        else:
            patch_size = [(size + (16 - size % 16) % 16) // 16 for size in args.shapelist]
            self.padding = [(16 - size % 16) % 16 for size in args.shapelist]

        # ==========================================
        # 3. U-Net 核心 (Multiscale modules)
        # ==========================================
        dim = len(patch_size)
        num_stages = 4 
        
        # 使用工厂模式极简实例化 U-Net
        self.encoder = OneEncoder(
            style=f"UNetEncoder{dim}D",
            in_channels=args.n_hidden,
            base_channels=args.n_hidden,
            num_stages=num_stages,
            bilinear=bilinear,
            normtype=normtype
        )
        
        self.decoder = OneDecoder(
            style=f"UNetDecoder{dim}D",
            base_channels=args.n_hidden,
            num_stages=num_stages,
            bilinear=bilinear,
            normtype=normtype
        )
        
        self.outc = OneHead(
            style=f"UNetHead{dim}D",
            in_channels=args.n_hidden,
            out_channels=args.n_hidden
        )

        # 最终投影
        self.fc1 = nn.Linear(args.n_hidden, args.n_hidden)
        self.fc2 = nn.Linear(args.n_hidden, args.out_dim)

    def multiscale(self, x):
        """完全解耦的 U-Net 前向计算"""
        # 提取多尺度特征
        features = self.encoder(x)
        # 融合上采样
        x = self.decoder(features)
        # 预测头输出
        return self.outc(x)

    def structured_geo(self, x, fx, T=None):
        B, N, _ = x.shape
        if self.args.unified_pos:
            x = self.pos.repeat(x.shape[0], 1, 1)
            
        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)

        if T is not None:
            Time_emb = timestep_embedding(T, self.args.n_hidden).repeat(1, x.shape[1], 1)
            Time_emb = self.time_fc(Time_emb)
            fx = fx + Time_emb
            
        x = fx.permute(0, 2, 1).reshape(B, self.args.n_hidden, *self.args.shapelist)
        
        # Padding
        if not all(item == 0 for item in self.padding):
            if len(self.args.shapelist) == 2:
                x = F.pad(x, [0, self.padding[1], 0, self.padding[0]])
            elif len(self.args.shapelist) == 3:
                x = F.pad(x, [0, self.padding[2], 0, self.padding[1], 0, self.padding[0]])
                
        # 极简调用多尺度 U-Net
        x = self.multiscale(x)
        
        # Unpadding
        if not all(item == 0 for item in self.padding):
            if len(self.args.shapelist) == 2:
                x = x[..., : -self.padding[0], : -self.padding[1]]
            elif len(self.args.shapelist) == 3:
                x = x[..., : -self.padding[0], : -self.padding[1], : -self.padding[2]]
                
        x = x.reshape(B, self.args.n_hidden, -1).permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def unstructured_geo(self, x, fx, T=None):
        original_pos = x
        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)

        if T is not None:
            Time_emb = timestep_embedding(T, self.args.n_hidden).repeat(1, x.shape[1], 1)
            Time_emb = self.time_fc(Time_emb)
            fx = fx + Time_emb

        # GeoFNO: 坐标映射与入场变换
        x = self.fftproject_in(
            fx.permute(0, 2, 1), x_in=original_pos, iphi=self.iphi, code=None
        )
        
        # 极简调用多尺度 U-Net
        x = self.multiscale(x)
        
        # GeoFNO: 出场逆变换
        x = self.fftproject_out(
            x, x_out=original_pos, iphi=self.iphi, code=None
        ).permute(0, 2, 1)
        
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def forward(self, x, fx, T=None, geo=None):
        if self.args.geotype == "unstructured":
            return self.unstructured_geo(x, fx, T)
        else:
            return self.structured_geo(x, fx, T)