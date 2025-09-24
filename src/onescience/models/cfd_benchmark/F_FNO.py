import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from timm.layers import trunc_normal_
from onescience.models.layers.Basic import MLP
from onescience.models.layers.Embedding import timestep_embedding, unified_pos_embedding
from onescience.models.layers.FFNO_Layers import (
    SpectralConv1d,
    SpectralConv2d,
    SpectralConv3d,
)
from onescience.models.layers.GeoFNO_Projection import SpectralConv2d_IrregularGeo, IPHI

BlockList = [None, SpectralConv1d, SpectralConv2d, SpectralConv3d]
ConvList = [None, nn.Conv1d, nn.Conv2d, nn.Conv3d]


class Model(nn.Module):
    def __init__(self, args, device, s1=96, s2=96):
        super(Model, self).__init__()
        self.__name__ = "F-FNO"
        self.args = args
        ## embedding
        if (
            args.unified_pos and args.geotype != "unstructured"
        ):  # only for structured mesh
            self.pos = unified_pos_embedding(args.shapelist, args.ref, device=device)
            self.preprocess = MLP(
                args.fun_dim + args.ref ** len(args.shapelist),
                args.n_hidden * 2,
                args.n_hidden,
                n_layers=0,
                res=False,
                act=args.act,
            )
        else:
            self.preprocess = MLP(
                args.fun_dim + args.space_dim,
                args.n_hidden * 2,
                args.n_hidden,
                n_layers=0,
                res=False,
                act=args.act,
            )
        if args.time_input:
            self.time_fc = nn.Sequential(
                nn.Linear(args.n_hidden, args.n_hidden),
                nn.SiLU(),
                nn.Linear(args.n_hidden, args.n_hidden),
            )
        # geometry projection
        if self.args.geotype == "unstructured":
            self.fftproject_in = SpectralConv2d_IrregularGeo(
                args.n_hidden, args.n_hidden, args.modes, args.modes, s1, s2
            )
            self.fftproject_out = SpectralConv2d_IrregularGeo(
                args.n_hidden, args.n_hidden, args.modes, args.modes, s1, s2
            )
            self.iphi = IPHI()
            self.padding = [(16 - size % 16) % 16 for size in [s1, s2]]
        else:
            self.padding = [(16 - size % 16) % 16 for size in args.shapelist]

        self.spectral_layers = nn.ModuleList([])
        for _ in range(args.n_layers):
            self.spectral_layers.append(
                BlockList[len(self.padding)](
                    args.n_hidden,
                    args.n_hidden,
                    *[args.modes for _ in range(len(self.padding))],
                )
            )
        # projectors
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
            Time_emb = timestep_embedding(T, self.args.n_hidden).repeat(
                1, x.shape[1], 1
            )
            Time_emb = self.time_fc(Time_emb)
            fx = fx + Time_emb
        x = fx.permute(0, 2, 1).reshape(B, self.args.n_hidden, *self.args.shapelist)
        if not all(item == 0 for item in self.padding):
            if len(self.args.shapelist) == 2:
                x = F.pad(x, [0, self.padding[1], 0, self.padding[0]])
            elif len(self.args.shapelist) == 3:
                x = F.pad(
                    x, [0, self.padding[2], 0, self.padding[1], 0, self.padding[0]]
                )

        for i in range(self.args.n_layers):
            x = x + self.spectral_layers[i](x)

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
            Time_emb = timestep_embedding(T, self.args.n_hidden).repeat(
                1, x.shape[1], 1
            )
            Time_emb = self.time_fc(Time_emb)
            fx = fx + Time_emb

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
