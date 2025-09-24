import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from timm.layers import trunc_normal_
from onescience.models.layers.Basic import MLP
from onescience.models.layers.Embedding import timestep_embedding, unified_pos_embedding
from onescience.models.layers.UNet_Blocks import (
    DoubleConv1D,
    Down1D,
    Up1D,
    OutConv1D,
    DoubleConv2D,
    Down2D,
    Up2D,
    OutConv2D,
    DoubleConv3D,
    Down3D,
    Up3D,
    OutConv3D,
)
from onescience.models.layers.GeoFNO_Projection import SpectralConv2d_IrregularGeo, IPHI

ConvList = [None, DoubleConv1D, DoubleConv2D, DoubleConv3D]
DownList = [None, Down1D, Down2D, Down3D]
UpList = [None, Up1D, Up2D, Up3D]
OutList = [None, OutConv1D, OutConv2D, OutConv3D]


class Model(nn.Module):
    def __init__(self, args, device, bilinear=True, s1=96, s2=96):
        super(Model, self).__init__()
        self.__name__ = "U-Net"
        self.args = args
        if args.task == "steady":
            normtype = "bn"
        else:
            normtype = (
                "in"  # when conducting dynamic tasks, use instance norm for stability
            )
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
            patch_size = [(size + (16 - size % 16) % 16) // 16 for size in [s1, s2]]
            self.padding = [(16 - size % 16) % 16 for size in [s1, s2]]
        else:
            patch_size = [
                (size + (16 - size % 16) % 16) // 16 for size in args.shapelist
            ]
            self.padding = [(16 - size % 16) % 16 for size in args.shapelist]
        # multiscale modules
        self.inc = ConvList[len(patch_size)](
            args.n_hidden, args.n_hidden, normtype=normtype
        )
        self.down1 = DownList[len(patch_size)](
            args.n_hidden, args.n_hidden * 2, normtype=normtype
        )
        self.down2 = DownList[len(patch_size)](
            args.n_hidden * 2, args.n_hidden * 4, normtype=normtype
        )
        self.down3 = DownList[len(patch_size)](
            args.n_hidden * 4, args.n_hidden * 8, normtype=normtype
        )
        factor = 2 if bilinear else 1
        self.down4 = DownList[len(patch_size)](
            args.n_hidden * 8, args.n_hidden * 16 // factor, normtype=normtype
        )
        self.up1 = UpList[len(patch_size)](
            args.n_hidden * 16, args.n_hidden * 8 // factor, bilinear, normtype=normtype
        )
        self.up2 = UpList[len(patch_size)](
            args.n_hidden * 8, args.n_hidden * 4 // factor, bilinear, normtype=normtype
        )
        self.up3 = UpList[len(patch_size)](
            args.n_hidden * 4, args.n_hidden * 2 // factor, bilinear, normtype=normtype
        )
        self.up4 = UpList[len(patch_size)](
            args.n_hidden * 2, args.n_hidden, bilinear, normtype=normtype
        )
        self.outc = OutList[len(patch_size)](args.n_hidden, args.n_hidden)
        # projectors
        self.fc1 = nn.Linear(args.n_hidden, args.n_hidden)
        self.fc2 = nn.Linear(args.n_hidden, args.out_dim)

    def multiscale(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

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
        x = self.multiscale(x)  ## U-Net
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
        x = self.multiscale(x)  ## U-Net
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
