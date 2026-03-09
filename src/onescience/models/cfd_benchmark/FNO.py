import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from onescience.modules import OneMlp
from onescience.modules import OneFourier
# IPHI 作为辅助映射网络，直接导入
from onescience.modules.fourier.geo_spectral import IPHI
from onescience.modules.embedding import timestep_embedding, unified_pos_embedding

ConvList = [None, nn.Conv1d, nn.Conv2d, nn.Conv3d]


class Model(nn.Module):
    """
    傅里叶神经算子 (Fourier Neural Operator, FNO)。
    支持 1D/2D/3D 结构化网格，以及基于 Geo-FNO 的非结构化网格。
    """
    def __init__(self, args, device, s1=96, s2=96):
        super(Model, self).__init__()
        self.__name__ = "FNO"
        self.args = args
        
        # ==========================================
        # 1. Embedding & Preprocess
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
            n_layers=0, # 如果 OneMlp 支持这个参数来控制层数
            res=False   # 如果 OneMlp 支持残差控制
        )

        if args.time_input:
            self.time_fc = nn.Sequential(
                nn.Linear(args.n_hidden, args.n_hidden),
                nn.SiLU(),
                nn.Linear(args.n_hidden, args.n_hidden),
            )

        # ==========================================
        # 2. Geometry Projection (GeoFNO 特有)
        # ==========================================
        if self.args.geotype == "unstructured":
            # 明确传入具体参数
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
            self.padding = [(16 - size % 16) % 16 for size in [s1, s2]]
        else:
            self.padding = [(16 - size % 16) % 16 for size in args.shapelist]

        # ==========================================
        # 3. FNO Blocks (显式参数实例化)
        # ==========================================
        dim = len(self.padding)
        
        if dim == 1:
            self.conv0 = OneFourier(style="FNOSpectralConv1d", in_channels=args.n_hidden, out_channels=args.n_hidden, modes1=args.modes)
            self.conv1 = OneFourier(style="FNOSpectralConv1d", in_channels=args.n_hidden, out_channels=args.n_hidden, modes1=args.modes)
            self.conv2 = OneFourier(style="FNOSpectralConv1d", in_channels=args.n_hidden, out_channels=args.n_hidden, modes1=args.modes)
            self.conv3 = OneFourier(style="FNOSpectralConv1d", in_channels=args.n_hidden, out_channels=args.n_hidden, modes1=args.modes)
        elif dim == 2:
            self.conv0 = OneFourier(style="FNOSpectralConv2d", in_channels=args.n_hidden, out_channels=args.n_hidden, modes1=args.modes, modes2=args.modes)
            self.conv1 = OneFourier(style="FNOSpectralConv2d", in_channels=args.n_hidden, out_channels=args.n_hidden, modes1=args.modes, modes2=args.modes)
            self.conv2 = OneFourier(style="FNOSpectralConv2d", in_channels=args.n_hidden, out_channels=args.n_hidden, modes1=args.modes, modes2=args.modes)
            self.conv3 = OneFourier(style="FNOSpectralConv2d", in_channels=args.n_hidden, out_channels=args.n_hidden, modes1=args.modes, modes2=args.modes)
        elif dim == 3:
            self.conv0 = OneFourier(style="FNOSpectralConv3d", in_channels=args.n_hidden, out_channels=args.n_hidden, modes1=args.modes, modes2=args.modes, modes3=args.modes)
            self.conv1 = OneFourier(style="FNOSpectralConv3d", in_channels=args.n_hidden, out_channels=args.n_hidden, modes1=args.modes, modes2=args.modes, modes3=args.modes)
            self.conv2 = OneFourier(style="FNOSpectralConv3d", in_channels=args.n_hidden, out_channels=args.n_hidden, modes1=args.modes, modes2=args.modes, modes3=args.modes)
            self.conv3 = OneFourier(style="FNOSpectralConv3d", in_channels=args.n_hidden, out_channels=args.n_hidden, modes1=args.modes, modes2=args.modes, modes3=args.modes)
        else:
            raise ValueError(f"Unsupported dimension: {dim}. Only 1D, 2D, and 3D are supported.")

        # 对应的 1x1 卷积通道混合层
        self.w0 = ConvList[dim](args.n_hidden, args.n_hidden, 1)
        self.w1 = ConvList[dim](args.n_hidden, args.n_hidden, 1)
        self.w2 = ConvList[dim](args.n_hidden, args.n_hidden, 1)
        self.w3 = ConvList[dim](args.n_hidden, args.n_hidden, 1)
        
        # ==========================================
        # 4. Projectors (输出层)
        # ==========================================
        self.fc1 = nn.Linear(args.n_hidden, args.n_hidden)
        self.fc2 = nn.Linear(args.n_hidden, args.out_dim)

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
                
        # Spectral Convs + Res connections
        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))
        x = self.conv3(x) + self.w3(x)

        # Unpadding
        if not all(item == 0 for item in self.padding):
            if len(self.args.shapelist) == 2:
                x = x[..., : -self.padding[0], : -self.padding[1]]
            elif len(self.args.shapelist) == 3:
                x = x[..., : -self.padding[0], : -self.padding[1], : -self.padding[2]]
                
        x = x.reshape(B, self.args.n_hidden, -1).permute(0, 2, 1)
        x = F.gelu(self.fc1(x))
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

        # 透传参数到 GeoSpectralConv2d
        x = self.fftproject_in(
            fx.permute(0, 2, 1), x_in=original_pos, iphi=self.iphi, code=None
        )

        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))
        x = self.conv3(x) + self.w3(x)

        # 透传参数到 GeoSpectralConv2d
        x = self.fftproject_out(
            x, x_out=original_pos, iphi=self.iphi, code=None
        ).permute(0, 2, 1)
        
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x

    def forward(self, x, fx, T=None, geo=None):
        if self.args.geotype == "unstructured":
            return self.unstructured_geo(x, fx, T)
        else:
            return self.structured_geo(x, fx, T)