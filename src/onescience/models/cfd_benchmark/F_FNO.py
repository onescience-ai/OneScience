import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from onescience.modules import OneFourier
from onescience.modules.mlp import StandardMLP
from onescience.modules.embedding import timestep_embedding, unified_pos_embedding
from onescience.modules.fourier.geo_spectral import IPHI

class Model(nn.Module):
    """
    Factorized Fourier Neural Operator (F-FNO) 模型。
    """
    def __init__(self, args, device, s1=96, s2=96):
        super(Model, self).__init__()
        self.__name__ = "F-FNO"
        self.args = args
        self.device = device
        
        # 1. Embedding & Preprocessing
        input_dim = args.fun_dim
        
        if args.unified_pos and args.geotype != "unstructured":
            self.pos = unified_pos_embedding(args.shapelist, args.ref, device=device)
            input_dim += args.ref ** len(args.shapelist)
        else:
            input_dim += args.space_dim

        self.preprocess = StandardMLP(
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

        # 2. Geometry Projection & Spectral Layers
        self.spectral_layers = nn.ModuleList([])
        
        if self.args.geotype == "unstructured":
            # --- 非结构化网格路径 (GeoFNO) ---
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
            
            # 中间层使用 FFNO
            spectral_style = "FFNOSpectralConv2d"
            conv_args = {
                "in_dim": args.n_hidden,
                "out_dim": args.n_hidden,
                "modes_x": args.modes,
                "modes_y": args.modes
            }
            
        else:
            # --- 结构化网格路径 (FFNO) ---
            self.padding = [(16 - size % 16) % 16 for size in args.shapelist]
            
            dim = len(self.padding)
            spectral_style = f"FFNOSpectralConv{dim}d"
            
            conv_args = {
                "in_dim": args.n_hidden,
                "out_dim": args.n_hidden,
            }
            
            mode_names = ["modes_x", "modes_y", "modes_z"]
            for i in range(dim):
                if i < len(mode_names):
                    conv_args[mode_names[i]] = args.modes

        for _ in range(args.n_layers):
            self.spectral_layers.append(
                OneFourier(style=spectral_style, **conv_args)
            )

        # 3. Projectors (Decoder)
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
            # 【修复】使用广播机制，避免 repeat 导致的维度错误
            Time_emb = timestep_embedding(T, self.args.n_hidden) # (B, C)
            Time_emb = self.time_fc(Time_emb)
            if Time_emb.ndim == 2:
                Time_emb = Time_emb.unsqueeze(1) # (B, 1, C)
            fx = fx + Time_emb # Broadcasting: (B, N, C) + (B, 1, C)
            
        x = fx.permute(0, 2, 1).reshape(B, self.args.n_hidden, *self.args.shapelist)
        
        if not all(item == 0 for item in self.padding):
            pad_arg = []
            for p in reversed(self.padding):
                pad_arg.extend([0, p]) 
            x = F.pad(x, pad_arg)

        for i in range(self.args.n_layers):
            x = x + self.spectral_layers[i](x)

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
            # 【修复】使用广播机制，避免 repeat 导致的维度错误
            Time_emb = timestep_embedding(T, self.args.n_hidden) # (B, C)
            Time_emb = self.time_fc(Time_emb)
            if Time_emb.ndim == 2:
                Time_emb = Time_emb.unsqueeze(1) # (B, 1, C)
            fx = fx + Time_emb # Broadcasting: (B, N, C) + (B, 1, C)

        x = self.fftproject_in(
            fx.permute(0, 2, 1), x_in=original_pos, iphi=self.iphi, code=None
        )
        
        for i in range(self.args.n_layers):
            x = x + self.spectral_layers[i](x)
            
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