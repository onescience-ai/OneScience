
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- 引入模块工厂 ---
from onescience.modules import OneMlp, OneFourier
from onescience.modules.embedding import timestep_embedding, unified_pos_embedding
from onescience.modules.fourier.geo_spectral import IPHI

# --- 引入 U-Net 基础组件 (替代 UNet_Blocks) ---
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
    U-NO (U-Net Neural Operator) 模型。
    
    结合了 U-Net 的多尺度结构和 FNO 的谱卷积能力。
    在 U-Net 的每个 Encoder 和 Decoder 层级之间插入了 FNO Block (SpectralConv) 和 1x1 卷积残差。
    """
    def __init__(self, args, device, bilinear=True, s1=96, s2=96):
        super(Model, self).__init__()
        self.__name__ = "U_NO"
        self.args = args

        if args.task == "steady":
            normtype = "bn"
        else:
            normtype = "in"

        # 1. Embedding & Preprocessing
        # -----------------------------------------------------------
        input_dim = args.fun_dim
        if args.unified_pos and args.geotype != "unstructured":
            self.pos = unified_pos_embedding(args.shapelist, args.ref, device=device)
            input_dim += args.ref ** len(args.shapelist)
        else:
            input_dim += args.space_dim

        self.preprocess = OneMlp(
            style="StandardMLP",
            input_dim=input_dim,
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

        # 2. Geometry Projection (GeoFNO for unstructured)
        # -----------------------------------------------------------
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
            self.augmented_resolution = [s1, s2]
        else:
            patch_size = [
                (size + (16 - size % 16) % 16) // 16 for size in args.shapelist
            ]
            self.padding = [(16 - size % 16) % 16 for size in args.shapelist]
            self.augmented_resolution = [
                shape + padding for shape, padding in zip(args.shapelist, self.padding)
            ]

        dim = len(patch_size) 

        # 3. Multiscale U-Net Modules 
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

        # 4. FNO Blocks
        def get_fno_layer(in_c, out_c, res_list, divisor):
            modes_list = [
                max(1, min(args.modes, res // divisor))
                for res in res_list
            ]

            style = f"FNOSpectralConv{dim}d"
            kwargs = {
                "in_channels": in_c,
                "out_channels": out_c
            }
            mode_names = ["modes1", "modes2", "modes3"]
            for i, m in enumerate(modes_list):
                if i < len(mode_names):
                    kwargs[mode_names[i]] = m
            return OneFourier(style, **kwargs)

        # Down Path FNOs
        self.process1_down = get_fno_layer(args.n_hidden, args.n_hidden, self.augmented_resolution, 2)
        self.process2_down = get_fno_layer(args.n_hidden * 2, args.n_hidden * 2, self.augmented_resolution, 4)
        self.process3_down = get_fno_layer(args.n_hidden * 4, args.n_hidden * 4, self.augmented_resolution, 8)
        self.process4_down = get_fno_layer(args.n_hidden * 8, args.n_hidden * 8, self.augmented_resolution, 16)
        self.process5_down = get_fno_layer(args.n_hidden * 16 // factor, args.n_hidden * 16 // factor, self.augmented_resolution, 32)

        # Residual Weights (1x1 Conv)
        self.w1_down = ConvList[dim](args.n_hidden, args.n_hidden, 1) # kernel_size=1
        self.w2_down = ConvList[dim](args.n_hidden * 2, args.n_hidden * 2, 1)
        self.w3_down = ConvList[dim](args.n_hidden * 4, args.n_hidden * 4, 1)
        self.w4_down = ConvList[dim](args.n_hidden * 8, args.n_hidden * 8, 1)
        self.w5_down = ConvList[dim](args.n_hidden * 16 // factor, args.n_hidden * 16 // factor, 1)

        # Up Path FNOs
        self.process1_up = get_fno_layer(args.n_hidden, args.n_hidden, self.augmented_resolution, 2)
        self.process2_up = get_fno_layer(args.n_hidden * 2 // factor, args.n_hidden * 2 // factor, self.augmented_resolution, 4)
        self.process3_up = get_fno_layer(args.n_hidden * 4 // factor, args.n_hidden * 4 // factor, self.augmented_resolution, 8)
        self.process4_up = get_fno_layer(args.n_hidden * 8 // factor, args.n_hidden * 8 // factor, self.augmented_resolution, 16)
        self.process5_up = get_fno_layer(args.n_hidden * 16 // factor, args.n_hidden * 16 // factor, self.augmented_resolution, 32)

        self.w1_up = ConvList[dim](args.n_hidden, args.n_hidden, 1)
        self.w2_up = ConvList[dim](args.n_hidden * 2 // factor, args.n_hidden * 2 // factor, 1)
        self.w3_up = ConvList[dim](args.n_hidden * 4 // factor, args.n_hidden * 4 // factor, 1)
        self.w4_up = ConvList[dim](args.n_hidden * 8 // factor, args.n_hidden * 8 // factor, 1)
        self.w5_up = ConvList[dim](args.n_hidden * 16 // factor, args.n_hidden * 16 // factor, 1)

        # 5. Projectors
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
            fx = fx + Time_emb # Broadcast

        # Reshape to Grid (B, C, H, W...)
        x = fx.permute(0, 2, 1).reshape(B, self.args.n_hidden, *self.args.shapelist)
        
        # Padding
        if not all(item == 0 for item in self.padding):
            pad_arg = []
            for p in reversed(self.padding):
                pad_arg.extend([0, p])
            x = F.pad(x, pad_arg)

        # === U-NO Body ===
        # Level 1 Down
        x1 = self.inc(x)
        # FNO Processing + Residual
        x1 = F.gelu(self.process1_down(x1) + self.w1_down(x1))
        
        # Level 2 Down
        x2 = self.down1(x1)
        x2 = F.gelu(self.process2_down(x2) + self.w2_down(x2))
        
        # Level 3 Down
        x3 = self.down2(x2)
        x3 = F.gelu(self.process3_down(x3) + self.w3_down(x3))
        
        # Level 4 Down
        x4 = self.down3(x3)
        x4 = F.gelu(self.process4_down(x4) + self.w4_down(x4))
        
        # Level 5 (Bottleneck)
        x5 = self.down4(x4)
        x5 = F.gelu(self.process5_down(x5) + self.w5_down(x5))
        # Bottleneck Up Process
        x5 = F.gelu(self.process5_up(x5) + self.w5_up(x5))
        
        # Level 4 Up
        x = self.up1(x5, x4)
        x = F.gelu(self.process4_up(x) + self.w4_up(x))
        
        # Level 3 Up
        x = self.up2(x, x3)
        x = F.gelu(self.process3_up(x) + self.w3_up(x))
        
        # Level 2 Up
        x = self.up3(x, x2)
        x = F.gelu(self.process2_up(x) + self.w2_up(x))
        
        # Level 1 Up
        x = self.up4(x, x1)
        x = F.gelu(self.process1_up(x) + self.w1_up(x))
        
        x = self.outc(x)

        # Un-padding
        if not all(item == 0 for item in self.padding):
            if len(self.args.shapelist) == 1:
                x = x[..., : -self.padding[0]]
            elif len(self.args.shapelist) == 2:
                x = x[..., : -self.padding[0], : -self.padding[1]]
            elif len(self.args.shapelist) == 3:
                x = x[..., : -self.padding[0], : -self.padding[1], : -self.padding[2]]
        
        # Output Projection
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

        # Projection to Grid
        x = self.fftproject_in(
            fx.permute(0, 2, 1), x_in=original_pos, iphi=self.iphi, code=None
        )
        
        # === U-NO Body===
        x1 = self.inc(x)
        x1 = F.gelu(self.process1_down(x1) + self.w1_down(x1))
        
        x2 = self.down1(x1)
        x2 = F.gelu(self.process2_down(x2) + self.w2_down(x2))
        
        x3 = self.down2(x2)
        x3 = F.gelu(self.process3_down(x3) + self.w3_down(x3))
        
        x4 = self.down3(x3)
        x4 = F.gelu(self.process4_down(x4) + self.w4_down(x4))
        
        x5 = self.down4(x4)
        x5 = F.gelu(self.process5_down(x5) + self.w5_down(x5))
        x5 = F.gelu(self.process5_up(x5) + self.w5_up(x5))
        
        x = self.up1(x5, x4)
        x = F.gelu(self.process4_up(x) + self.w4_up(x))
        
        x = self.up2(x, x3)
        x = F.gelu(self.process3_up(x) + self.w3_up(x))
        
        x = self.up3(x, x2)
        x = F.gelu(self.process2_up(x) + self.w2_up(x))
        
        x = self.up4(x, x1)
        x = F.gelu(self.process1_up(x) + self.w1_up(x))
        
        x = self.outc(x)
        
        # Projection Back to Points
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