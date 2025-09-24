import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from timm.layers import trunc_normal_
from onescience.models.layers.Basic import MLP
from onescience.models.layers.Embedding import timestep_embedding, unified_pos_embedding
from onescience.models.layers.MWT_Layers import MWT_CZ1d, MWT_CZ2d, MWT_CZ3d
from onescience.models.layers.GeoFNO_Projection import SpectralConv2d_IrregularGeo, IPHI

BlockList = [None, MWT_CZ1d, MWT_CZ2d, MWT_CZ3d]
ConvList = [None, nn.Conv1d, nn.Conv2d, nn.Conv3d]


class Model(nn.Module):
    # this model requires H = W = Z and H, W, Z is the power of two
    def __init__(
        self, args, device, alpha=2, L=0, c=1, base="legendre", s1=128, s2=128
    ):
        super(Model, self).__init__()
        self.__name__ = "MWT"
        self.args = args
        self.k = args.mwt_k
        self.WMT_dim = c * self.k**2
        if args.geotype == "structured_1D":
            self.WMT_dim = c * self.k
        self.c = c
        self.s1 = s1
        self.s2 = s2
        ## embedding
        if (
            args.unified_pos and args.geotype != "unstructured"
        ):  # only for structured mesh
            self.pos = unified_pos_embedding(args.shapelist, args.ref, device=device)
            self.preprocess = MLP(
                args.fun_dim + args.ref ** len(args.shapelist),
                args.n_hidden * 2,
                self.WMT_dim,
                n_layers=0,
                res=False,
                act=args.act,
            )
        else:
            self.preprocess = MLP(
                args.fun_dim + args.space_dim,
                args.n_hidden * 2,
                self.WMT_dim,
                n_layers=0,
                res=False,
                act=args.act,
            )
        if args.time_input:
            self.time_fc = nn.Sequential(
                nn.Linear(self.WMT_dim, args.n_hidden),
                nn.SiLU(),
                nn.Linear(args.n_hidden, self.WMT_dim),
            )
        # geometry projection
        if self.args.geotype == "unstructured":
            self.fftproject_in = SpectralConv2d_IrregularGeo(
                self.WMT_dim, self.WMT_dim, args.modes, args.modes, s1, s2
            )
            self.fftproject_out = SpectralConv2d_IrregularGeo(
                self.WMT_dim, self.WMT_dim, args.modes, args.modes, s1, s2
            )
            self.iphi = IPHI()
            self.augmented_resolution = [s1, s2]
            self.padding = [(16 - size % 16) % 16 for size in [s1, s2]]
        else:
            target = 2 ** (math.ceil(np.log2(max(args.shapelist))))
            self.padding = [(target - size) for size in args.shapelist]
            self.augmented_resolution = [target for _ in range(len(self.padding))]
        self.spectral_layers = nn.ModuleList(
            [
                BlockList[len(self.padding)](k=self.k, alpha=alpha, L=L, c=c, base=base)
                for _ in range(args.n_layers)
            ]
        )
        # projectors
        self.fc1 = nn.Linear(self.WMT_dim, args.n_hidden)
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
            Time_emb = timestep_embedding(T, self.WMT_dim).repeat(1, x.shape[1], 1)
            Time_emb = self.time_fc(Time_emb)
            fx = fx + Time_emb
        x = fx.permute(0, 2, 1).reshape(B, self.WMT_dim, *self.args.shapelist)
        if not all(item == 0 for item in self.padding):
            if len(self.args.shapelist) == 2:
                x = F.pad(x, [0, self.padding[1], 0, self.padding[0]])
            elif len(self.args.shapelist) == 3:
                x = F.pad(
                    x, [0, self.padding[2], 0, self.padding[1], 0, self.padding[0]]
                )
        x = (
            x.reshape(B, self.WMT_dim, -1)
            .permute(0, 2, 1)
            .contiguous()
            .reshape(
                B,
                *self.augmented_resolution,
                self.c,
                self.k**2 if self.args.geotype != "structured_1D" else self.k
            )
        )
        for i in range(self.args.n_layers):
            x = self.spectral_layers[i](x)
            if i < self.args.n_layers - 1:
                x = F.gelu(x)
        x = (
            x.reshape(B, -1, self.WMT_dim)
            .permute(0, 2, 1)
            .contiguous()
            .reshape(B, self.WMT_dim, *self.augmented_resolution)
        )
        if not all(item == 0 for item in self.padding):
            if len(self.args.shapelist) == 2:
                x = x[..., : -self.padding[0], : -self.padding[1]]
            elif len(self.args.shapelist) == 3:
                x = x[..., : -self.padding[0], : -self.padding[1], : -self.padding[2]]
        x = x.reshape(B, self.WMT_dim, -1).permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def unstructured_geo(self, x, fx, T=None):
        B, N, _ = x.shape
        original_pos = x
        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)

        if T is not None:
            Time_emb = timestep_embedding(T, self.WMT_dim).repeat(1, x.shape[1], 1)
            Time_emb = self.time_fc(Time_emb)
            fx = fx + Time_emb

        x = self.fftproject_in(
            fx.permute(0, 2, 1), x_in=original_pos, iphi=self.iphi, code=None
        )
        x = (
            x.reshape(B, self.WMT_dim, -1)
            .permute(0, 2, 1)
            .contiguous()
            .reshape(B, *self.augmented_resolution, self.c, self.k**2)
        )
        for i in range(self.args.n_layers):
            x = self.spectral_layers[i](x)
            if i < self.args.n_layers - 1:
                x = F.gelu(x)
        x = (
            x.reshape(B, -1, self.WMT_dim)
            .permute(0, 2, 1)
            .contiguous()
            .reshape(B, self.WMT_dim, *self.augmented_resolution)
        )
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
