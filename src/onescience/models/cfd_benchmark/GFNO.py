import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from onescience.modules import OneMlp, OneFourier, OneEquivariant
from onescience.modules.embedding import timestep_embedding, unified_pos_embedding
from onescience.modules.fourier.geo_spectral import IPHI

class GNorm(nn.Module):
    def __init__(self, width, group_size):
        super().__init__()
        self.group_size = group_size
        self.norm = torch.nn.InstanceNorm3d(width)

    def forward(self, x):
        # x shape: (B, C*Group, H, W) -> (B, C, Group, H, W)
        x = x.view(x.shape[0], -1, self.group_size, x.shape[-2], x.shape[-1])
        x = self.norm(x)
        x = x.view(x.shape[0], -1, x.shape[-2], x.shape[-1])
        return x

class Model(nn.Module):
    """
    GFNO (Group Factorized Neural Operator) 模型。
    
    """
    def __init__(self, args, device, s1=96, s2=96):
        super(Model, self).__init__()
        self.__name__ = "GFNO"
        self.args = args
        self.device = device
        
        self.in_channels = args.fun_dim
        self.out_channels = args.out_dim
        self.modes = args.modes
        self.width = args.n_hidden

        reflection = False

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

        # 2. Time Embedding
        if args.time_input:
            self.time_fc = nn.Sequential(
                nn.Linear(args.n_hidden, args.n_hidden),
                nn.SiLU(),
                nn.Linear(args.n_hidden, args.n_hidden),
            )
        else:
            self.time_fc = None

        # 3. Geometry Projection & Output Heads
        if args.geotype == "unstructured":
            self.group_size = 4 * (1 + reflection)
            
            self.fftproject_in = OneFourier(
                style="GeoSpectralConv2d",
                in_channels=args.n_hidden,
                out_channels=args.n_hidden,
                modes1=args.modes,
                modes2=args.modes,
                s1=s1,
                s2=s2,
            )
            self.fftproject_out = OneFourier(
                style="GeoSpectralConv2d",
                in_channels=self.width * self.group_size,
                out_channels=self.width,
                modes1=args.modes,
                modes2=args.modes,
                s1=s1,
                s2=s2,
            )
            self.iphi = IPHI()
            
            grid_h, grid_w = s1, s2
            self.point_q = nn.Sequential(
                nn.Linear(self.width, self.width * 4),
                nn.GELU(),
                nn.Linear(self.width * 4, self.out_channels),
            )
        else:
            grid_h, grid_w = args.shapelist
            # 结构化输出头 (Explicit instantiation)
            self.q = OneMlp(
                style="GroupEquivariantMLP2d",
                in_channels=self.width,
                out_channels=self.out_channels,
                mid_channels=self.width * 4,
                reflection=reflection,
                last_layer=True,
            )

        self.padding = [(16 - size % 16) % 16 for size in [grid_h, grid_w]]

        # 4. GFNO Stem Layers
        # Lifting Layer
        self.p = OneEquivariant(
            style="GroupEquivariantConv2d",
            in_channels=args.n_hidden,
            out_channels=self.width,
            kernel_size=1,
            reflection=reflection,
            first_layer=True,
        )
        
        # Spectral Layers
        self.conv0 = OneFourier("GSpectralConv2d", in_channels=self.width, out_channels=self.width, modes=self.modes, reflection=reflection)
        self.conv1 = OneFourier("GSpectralConv2d", in_channels=self.width, out_channels=self.width, modes=self.modes, reflection=reflection)
        self.conv2 = OneFourier("GSpectralConv2d", in_channels=self.width, out_channels=self.width, modes=self.modes, reflection=reflection)
        self.conv3 = OneFourier("GSpectralConv2d", in_channels=self.width, out_channels=self.width, modes=self.modes, reflection=reflection)
        
        self.mlp0 = OneMlp(
            style="GroupEquivariantMLP2d",
            in_channels=self.width,
            out_channels=self.width,
            mid_channels=self.width,
            reflection=reflection
        )
        self.mlp1 = OneMlp(
            style="GroupEquivariantMLP2d",
            in_channels=self.width,
            out_channels=self.width,
            mid_channels=self.width,
            reflection=reflection
        )
        self.mlp2 = OneMlp(
            style="GroupEquivariantMLP2d",
            in_channels=self.width,
            out_channels=self.width,
            mid_channels=self.width,
            reflection=reflection
        )
        self.mlp3 = OneMlp(
            style="GroupEquivariantMLP2d",
            in_channels=self.width,
            out_channels=self.width,
            mid_channels=self.width,
            reflection=reflection
        )
        
        # Residual Weights (1x1 GConv via OneEquivariant)
        # 显式构建
        self.w0 = OneEquivariant(
            style="GroupEquivariantConv2d",
            in_channels=self.width,
            out_channels=self.width,
            kernel_size=1,
            reflection=reflection
        )
        self.w1 = OneEquivariant(
            style="GroupEquivariantConv2d",
            in_channels=self.width,
            out_channels=self.width,
            kernel_size=1,
            reflection=reflection
        )
        self.w2 = OneEquivariant(
            style="GroupEquivariantConv2d",
            in_channels=self.width,
            out_channels=self.width,
            kernel_size=1,
            reflection=reflection
        )
        self.w3 = OneEquivariant(
            style="GroupEquivariantConv2d",
            in_channels=self.width,
            out_channels=self.width,
            kernel_size=1,
            reflection=reflection
        )
        
        self.norm = GNorm(self.width, group_size=4 * (1 + reflection))

    def _gfno_stem_forward(self, x):
        # x: (B, C, H, W)
        x = self.p(x) # Lifting

        # 4 Layers
        x1 = self.norm(self.conv0(self.norm(x)))
        x1 = self.mlp0(x1)
        x = F.gelu(x1 + self.w0(x))
        
        x1 = self.norm(self.conv1(self.norm(x)))
        x1 = self.mlp1(x1)
        x = F.gelu(x1 + self.w1(x))
        
        x1 = self.norm(self.conv2(self.norm(x)))
        x1 = self.mlp2(x1)
        x = F.gelu(x1 + self.w2(x))
        
        x1 = self.norm(self.conv3(self.norm(x)))
        x1 = self.mlp3(x1)
        x = x1 + self.w3(x)
        return x

    def structured_geo(self, x, fx, T=None):
        B, N, _ = x.shape

        if self.args.unified_pos:
            pos = self.pos.repeat(B, 1, 1)
            feats = torch.cat([pos, fx], dim=-1) if fx is not None else pos
        else:
            feats = torch.cat([x, fx], dim=-1) if fx is not None else x

        feats = self.preprocess(feats) 

        if (T is not None) and (self.time_fc is not None):
            t_emb = timestep_embedding(T, self.args.n_hidden)
            t_emb = self.time_fc(t_emb)
            if t_emb.ndim == 2:
                t_emb = t_emb.unsqueeze(1)
            feats = feats + t_emb # Broadcast

        H, W = self.args.shapelist
        xg = feats.permute(0, 2, 1).reshape(B, self.args.n_hidden, H, W)

        if not all(p == 0 for p in self.padding):
            xg = F.pad(xg, [0, self.padding[1], 0, self.padding[0]])

        xg = self._gfno_stem_forward(xg)

        if not all(p == 0 for p in self.padding):
            xg = xg[..., : -self.padding[0], : -self.padding[1]]

        xg = self.q(xg) 
        out = xg.reshape(B, self.out_channels, -1).permute(0, 2, 1)
        return out

    def unstructured_geo(self, x, fx, T=None):
        B, N, _ = x.shape

        feats = torch.cat([x, fx], dim=-1) if fx is not None else x
        feats = self.preprocess(feats) 

        if (T is not None) and (self.time_fc is not None):
            t_emb = timestep_embedding(T, self.args.n_hidden)
            t_emb = self.time_fc(t_emb)
            if t_emb.ndim == 2:
                t_emb = t_emb.unsqueeze(1)
            feats = feats + t_emb

        xg = self.fftproject_in(
            feats.permute(0, 2, 1), x_in=x, iphi=self.iphi, code=None
        )

        xg = self._gfno_stem_forward(xg)

        xp = self.fftproject_out(xg, x_out=x, iphi=self.iphi, code=None).permute(
            0, 2, 1
        ) 

        out = self.point_q(xp) 
        return out

    def forward(self, x, fx=None, T=None, geo=None):
        if self.args.geotype == "unstructured":
            return self.unstructured_geo(x, fx, T)
        else:
            return self.structured_geo(x, fx, T)