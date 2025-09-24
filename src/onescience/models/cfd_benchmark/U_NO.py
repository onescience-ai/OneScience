import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from timm.layers import trunc_normal_
from onescience.models.layers.Basic import MLP
from onescience.models.layers.Embedding import timestep_embedding, unified_pos_embedding
from onescience.models.layers.FNO_Layers import (
    SpectralConv1d,
    SpectralConv2d,
    SpectralConv3d,
)
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
BlockList = [None, SpectralConv1d, SpectralConv2d, SpectralConv3d]


class Model(nn.Module):
    def __init__(self, args, device, bilinear=True, s1=96, s2=96):
        super(Model, self).__init__()
        self.__name__ = "U_NO"
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
            self.augmented_resolution = [s1, s2]
        else:
            patch_size = [
                (size + (16 - size % 16) % 16) // 16 for size in args.shapelist
            ]
            self.padding = [(16 - size % 16) % 16 for size in args.shapelist]
            self.augmented_resolution = [
                shape + padding for shape, padding in zip(args.shapelist, self.padding)
            ]

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
        # Down FNO
        self.process1_down = BlockList[len(patch_size)](
            args.n_hidden,
            args.n_hidden,
            *[
                max(1, min(args.modes, min(self.augmented_resolution) // 2))
                for _ in range(len(self.padding))
            ]
        )
        self.process2_down = BlockList[len(patch_size)](
            args.n_hidden * 2,
            args.n_hidden * 2,
            *[
                max(1, min(args.modes, min(self.augmented_resolution) // 4))
                for _ in range(len(self.padding))
            ]
        )
        self.process3_down = BlockList[len(patch_size)](
            args.n_hidden * 4,
            args.n_hidden * 4,
            *[
                max(1, min(args.modes, min(self.augmented_resolution) // 8))
                for _ in range(len(self.padding))
            ]
        )
        self.process4_down = BlockList[len(patch_size)](
            args.n_hidden * 8,
            args.n_hidden * 8,
            *[
                max(1, min(args.modes, min(self.augmented_resolution) // 16))
                for _ in range(len(self.padding))
            ]
        )
        self.process5_down = BlockList[len(patch_size)](
            args.n_hidden * 16 // factor,
            args.n_hidden * 16 // factor,
            *[
                max(1, min(args.modes, min(self.augmented_resolution) // 32))
                for _ in range(len(self.padding))
            ]
        )
        self.w1_down = ConvList[len(self.padding)](args.n_hidden, args.n_hidden, 1)
        self.w2_down = ConvList[len(self.padding)](
            args.n_hidden * 2, args.n_hidden * 2, 1
        )
        self.w3_down = ConvList[len(self.padding)](
            args.n_hidden * 4, args.n_hidden * 4, 1
        )
        self.w4_down = ConvList[len(self.padding)](
            args.n_hidden * 8, args.n_hidden * 8, 1
        )
        self.w5_down = ConvList[len(self.padding)](
            args.n_hidden * 16 // factor, args.n_hidden * 16 // factor, 1
        )
        # Up FNO
        self.process1_up = BlockList[len(patch_size)](
            args.n_hidden,
            args.n_hidden,
            *[
                max(1, min(args.modes, min(self.augmented_resolution) // 2))
                for _ in range(len(self.padding))
            ]
        )
        self.process2_up = BlockList[len(patch_size)](
            args.n_hidden * 2 // factor,
            args.n_hidden * 2 // factor,
            *[
                max(1, min(args.modes, min(self.augmented_resolution) // 4))
                for _ in range(len(self.padding))
            ]
        )
        self.process3_up = BlockList[len(patch_size)](
            args.n_hidden * 4 // factor,
            args.n_hidden * 4 // factor,
            *[
                max(1, min(args.modes, min(self.augmented_resolution) // 8))
                for _ in range(len(self.padding))
            ]
        )
        self.process4_up = BlockList[len(patch_size)](
            args.n_hidden * 8 // factor,
            args.n_hidden * 8 // factor,
            *[
                max(1, min(args.modes, min(self.augmented_resolution) // 16))
                for _ in range(len(self.padding))
            ]
        )
        self.process5_up = BlockList[len(patch_size)](
            args.n_hidden * 16 // factor,
            args.n_hidden * 16 // factor,
            *[
                max(1, min(args.modes, min(self.augmented_resolution) // 32))
                for _ in range(len(self.padding))
            ]
        )
        self.w1_up = ConvList[len(self.padding)](args.n_hidden, args.n_hidden, 1)
        self.w2_up = ConvList[len(self.padding)](
            args.n_hidden * 2 // factor, args.n_hidden * 2 // factor, 1
        )
        self.w3_up = ConvList[len(self.padding)](
            args.n_hidden * 4 // factor, args.n_hidden * 4 // factor, 1
        )
        self.w4_up = ConvList[len(self.padding)](
            args.n_hidden * 8 // factor, args.n_hidden * 8 // factor, 1
        )
        self.w5_up = ConvList[len(self.padding)](
            args.n_hidden * 16 // factor, args.n_hidden * 16 // factor, 1
        )
        # projectors
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
