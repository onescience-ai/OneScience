import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from onescience.modules import OneMlp, OneFourier
from onescience.modules.embedding import timestep_embedding, unified_pos_embedding
from onescience.modules.fourier.geo_spectral import IPHI

from .U_Net import Model as U_Net

class Model(nn.Module):
    """
    U-FNO 模型。
    
    结合了 U-Net (用于多尺度特征提取) 和 FNO (用于全局谱特征提取)。
    U-Net 作为 FNO 层的并联分支，增强了局部特征捕捉能力。

    """
    def __init__(self, args, device, s1=96, s2=96):
        super(Model, self).__init__()
        self.__name__ = "U-FNO"
        self.args = args
        self.device = device

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

        # 2. Geometry Projection & Padding Logic
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
            
            self.padding = [(16 - size % 16) % 16 for size in [s1, s2]]
        else:
            self.padding = [(16 - size % 16) % 16 for size in args.shapelist]

        # 3. FNO Blocks
        dim = len(self.padding)
        
        # 辅助函数：构建 FNO 参数 
        def get_fno_layer(in_c, out_c):
            style = f"FNOSpectralConv{dim}d"
            kwargs = {
                "in_channels": in_c,
                "out_channels": out_c
            }
            # 动态添加 modes1, modes2, modes3
            mode_names = ["modes1", "modes2", "modes3"]
            for i in range(dim):
                if i < len(mode_names):
                    kwargs[mode_names[i]] = args.modes
            return OneFourier(style, **kwargs)

        self.conv0 = get_fno_layer(args.n_hidden, args.n_hidden)
        self.conv1 = get_fno_layer(args.n_hidden, args.n_hidden)
        self.conv2 = get_fno_layer(args.n_hidden, args.n_hidden)
        self.conv3 = get_fno_layer(args.n_hidden, args.n_hidden)

        ConvClass = [None, nn.Conv1d, nn.Conv2d, nn.Conv3d][dim]
        self.w0 = ConvClass(args.n_hidden, args.n_hidden, 1)
        self.w1 = ConvClass(args.n_hidden, args.n_hidden, 1)
        self.w2 = ConvClass(args.n_hidden, args.n_hidden, 1)
        self.w3 = ConvClass(args.n_hidden, args.n_hidden, 1)

        # 4. U-Net Branches (Parallel)
        # -----------------------------------------------------------
        self.u_net2 = U_Net(args, device)
        self.u_net3 = U_Net(args, device)

        # 5. Projectors
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
            Time_emb = timestep_embedding(T, self.args.n_hidden) # (B, C)
            Time_emb = self.time_fc(Time_emb)
            if Time_emb.ndim == 2:
                Time_emb = Time_emb.unsqueeze(1) # (B, 1, C)
            fx = fx + Time_emb

        x = fx.permute(0, 2, 1).reshape(B, self.args.n_hidden, *self.args.shapelist)
        
        if not all(item == 0 for item in self.padding):
            pad_arg = []
            for p in reversed(self.padding):
                pad_arg.extend([0, p])
            x = F.pad(x, pad_arg)

        # Layer 0
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        # Layer 1
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        # Layer 2 (with U-Net)
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x3 = self.u_net2.multiscale(x)
        x = x1 + x2 + x3
        x = F.gelu(x)

        # Layer 3 (with U-Net)
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x3 = self.u_net3.multiscale(x)
        x = x1 + x2 + x3

        if not all(item == 0 for item in self.padding):
            if len(self.args.shapelist) == 1:
                x = x[..., :-self.padding[0]]
            elif len(self.args.shapelist) == 2:
                x = x[..., :-self.padding[0], :-self.padding[1]]
            elif len(self.args.shapelist) == 3:
                x = x[..., :-self.padding[0], :-self.padding[1], :-self.padding[2]]
        
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

        # Layer 0
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        # Layer 1
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        # Layer 2
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x3 = self.u_net2.multiscale(x)
        x = x1 + x2 + x3
        x = F.gelu(x)

        # Layer 3
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x3 = self.u_net3.multiscale(x)
        x = x1 + x2 + x3

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