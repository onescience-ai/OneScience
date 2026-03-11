
import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from onescience.modules import OneMlp, OneFourier, OneTransformer
from onescience.modules.embedding import timestep_embedding, unified_pos_embedding
from onescience.modules.fourier.geo_spectral import IPHI
from onescience.modules.layer.unet_layer import (
    DoubleConv1D, Down1D, Up1D, OutConv1D,
    DoubleConv2D, Down2D, Up2D, OutConv2D,
    DoubleConv3D, Down3D, Up3D, OutConv3D,
)

ConvList = [None, DoubleConv1D, DoubleConv2D, DoubleConv3D]
DownList = [None, Down1D, Down2D, Down3D]
UpList = [None, Up1D, Up2D, Up3D]
OutList = [None, OutConv1D, OutConv2D, OutConv3D]

class Model(nn.Module):
    """
    LSM (Latent Spectral Model) 模型。
    结合了 U-Net 的多尺度特征提取与 Neural Spectral Block (Latent Transformer) 的全局谱处理能力。
    """
    def __init__(
        self, args, device, bilinear=True, num_token=4, num_basis=12, s1=96, s2=96
    ):
        super(Model, self).__init__()
        self.__name__ = "LSM"
        self.args = args
        
        if args.task == "steady":
            normtype = "bn"
        else:
            normtype = "in"  

        # 1. Embedding & Preprocessing 
        if args.unified_pos and args.geotype != "unstructured":
            self.pos = unified_pos_embedding(args.shapelist, args.ref, device=device)
            in_dim = args.fun_dim + args.ref ** len(args.shapelist)
        else:
            in_dim = args.fun_dim + args.space_dim

        self.preprocess = OneMlp(
            style="StandardMLP",
            input_dim=in_dim,
            output_dim=args.n_hidden,
            hidden_dims=[args.n_hidden * 2],
            activation=args.act,
            use_bias=True
        )

        if args.time_input:
            self.time_fc = nn.Sequential(
                nn.Linear(args.n_hidden, args.n_hidden),
                nn.SiLU(),
                nn.Linear(args.n_hidden, args.n_hidden),
            )
        else:
            self.time_fc = None

        # 2. Geometry Projection 
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
            patch_size = [
                (size + (16 - size % 16) % 16) // 16 for size in args.shapelist
            ]
            self.padding = [(16 - size % 16) % 16 for size in args.shapelist]

        # 3. Multiscale U-Net Modules
        # -----------------------------------------------------------
        dim = len(patch_size)
        
        self.inc = ConvList[dim](args.n_hidden, args.n_hidden, normtype=normtype)
        self.down1 = DownList[dim](args.n_hidden, args.n_hidden * 2, normtype=normtype)
        self.down2 = DownList[dim](args.n_hidden * 2, args.n_hidden * 4, normtype=normtype)
        self.down3 = DownList[dim](args.n_hidden * 4, args.n_hidden * 8, normtype=normtype)
        
        factor = 2 if bilinear else 1
        self.down4 = DownList[dim](args.n_hidden * 8, args.n_hidden * 16 // factor, normtype=normtype)
        
        self.up1 = UpList[dim](args.n_hidden * 16, args.n_hidden * 8 // factor, bilinear, normtype=normtype)
        self.up2 = UpList[dim](args.n_hidden * 8, args.n_hidden * 4 // factor, bilinear, normtype=normtype)
        self.up3 = UpList[dim](args.n_hidden * 4, args.n_hidden * 2 // factor, bilinear, normtype=normtype)
        self.up4 = UpList[dim](args.n_hidden * 2, args.n_hidden, bilinear, normtype=normtype)
        self.outc = OutList[dim](args.n_hidden, args.n_hidden)

        # 4. Patchified Neural Spectral Blocks 
        style_name = f"NeuralSpectralBlock{dim}D"
        
        self.process1 = OneTransformer(
            style=style_name,
            width=args.n_hidden,
            num_basis=num_basis,
            patch_size=patch_size,
            num_token=num_token,
            n_heads=args.n_heads
        )
        self.process2 = OneTransformer(
            style=style_name,
            width=args.n_hidden * 2,
            num_basis=num_basis,
            patch_size=patch_size,
            num_token=num_token,
            n_heads=args.n_heads
        )
        self.process3 = OneTransformer(
            style=style_name,
            width=args.n_hidden * 4,
            num_basis=num_basis,
            patch_size=patch_size,
            num_token=num_token,
            n_heads=args.n_heads
        )
        self.process4 = OneTransformer(
            style=style_name,
            width=args.n_hidden * 8,
            num_basis=num_basis,
            patch_size=patch_size,
            num_token=num_token,
            n_heads=args.n_heads
        )
        self.process5 = OneTransformer(
            style=style_name,
            width=args.n_hidden * 16 // factor,
            num_basis=num_basis,
            patch_size=patch_size,
            num_token=num_token,
            n_heads=args.n_heads
        )

        # 5. Projectors
        # -----------------------------------------------------------
        self.fc1 = nn.Linear(args.n_hidden, args.n_hidden * 2)
        self.fc2 = nn.Linear(args.n_hidden * 2, args.out_dim)

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
            Time_emb = timestep_embedding(T, self.args.n_hidden)
            Time_emb = self.time_fc(Time_emb)
            if Time_emb.ndim == 2:
                Time_emb = Time_emb.unsqueeze(1)
            fx = fx + Time_emb
            
        x = fx.permute(0, 2, 1).reshape(B, self.args.n_hidden, *self.args.shapelist)
        
        if not all(item == 0 for item in self.padding):
            pad_arg = []
            for p in reversed(self.padding):
                pad_arg.extend([0, p])
            x = F.pad(x, pad_arg)

        # LSM 核心处理流
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(self.process5(x5), self.process4(x4))
        x = self.up2(x, self.process3(x3))
        x = self.up3(x, self.process2(x2))
        x = self.up4(x, self.process1(x1))
        x = self.outc(x)

        if not all(item == 0 for item in self.padding):
            if len(self.args.shapelist) == 1:
                x = x[..., : -self.padding[0]]
            elif len(self.args.shapelist) == 2:
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
            Time_emb = timestep_embedding(T, self.args.n_hidden)
            Time_emb = self.time_fc(Time_emb)
            if Time_emb.ndim == 2:
                Time_emb = Time_emb.unsqueeze(1)
            fx = fx + Time_emb

        x = self.fftproject_in(
            fx.permute(0, 2, 1), x_in=original_pos, iphi=self.iphi, code=None
        )
        
        # LSM 核心处理流
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(self.process5(x5), self.process4(x4))
        x = self.up2(x, self.process3(x3))
        x = self.up3(x, self.process2(x2))
        x = self.up4(x, self.process1(x1))
        x = self.outc(x)
        
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