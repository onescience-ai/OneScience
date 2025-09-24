"""
Taken and modified from AIRS/OpenPDE
https://github.com/divelab/AIRS/blob/main/OpenPDE/G-FNO/models/GFNO.py
"""

import torch.nn.functional as F
import torch
import torch.nn as nn
import math
from onescience.models.layers.Embedding import timestep_embedding, unified_pos_embedding
from onescience.models.layers.Basic import MLP
from onescience.models.layers.GeoFNO_Projection import SpectralConv2d_IrregularGeo, IPHI

# ----------------------------------------------------------------------------------------------------------------------
# GFNO2d
# ----------------------------------------------------------------------------------------------------------------------


class GConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        bias=True,
        first_layer=False,
        last_layer=False,
        spectral=False,
        Hermitian=False,
        reflection=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reflection = reflection
        self.rt_group_size = 4
        self.group_size = self.rt_group_size * (1 + reflection)
        assert kernel_size % 2 == 1, "kernel size must be odd"
        dtype = torch.cfloat if spectral else torch.float
        self.kernel_size_Y = kernel_size
        self.kernel_size_X = kernel_size // 2 + 1 if Hermitian else kernel_size
        self.Hermitian = Hermitian
        if first_layer or last_layer:
            self.W = nn.Parameter(
                torch.empty(
                    out_channels,
                    1,
                    in_channels,
                    self.kernel_size_Y,
                    self.kernel_size_X,
                    dtype=dtype,
                )
            )
        else:
            if self.Hermitian:
                self.W = nn.ParameterDict(
                    {
                        "y0_modes": torch.nn.Parameter(
                            torch.empty(
                                out_channels,
                                1,
                                in_channels,
                                self.group_size,
                                self.kernel_size_X - 1,
                                1,
                                dtype=dtype,
                            )
                        ),
                        "yposx_modes": torch.nn.Parameter(
                            torch.empty(
                                out_channels,
                                1,
                                in_channels,
                                self.group_size,
                                self.kernel_size_Y,
                                self.kernel_size_X - 1,
                                dtype=dtype,
                            )
                        ),
                        "00_modes": torch.nn.Parameter(
                            torch.empty(
                                out_channels,
                                1,
                                in_channels,
                                self.group_size,
                                1,
                                1,
                                dtype=torch.float,
                            )
                        ),
                    }
                )
            else:
                self.W = nn.Parameter(
                    torch.empty(
                        out_channels,
                        1,
                        in_channels,
                        self.group_size,
                        self.kernel_size_Y,
                        self.kernel_size_X,
                        dtype=dtype,
                    )
                )
        self.first_layer = first_layer
        self.last_layer = last_layer
        self.B = nn.Parameter(torch.empty(1, out_channels, 1, 1)) if bias else None
        self.eval_build = True
        self.reset_parameters()
        self.get_weight()

    def reset_parameters(self):
        if self.Hermitian:
            for v in self.W.values():
                nn.init.kaiming_uniform_(v, a=math.sqrt(5))
        else:
            nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        if self.B is not None:
            nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))

    def get_weight(self):

        if self.training:
            self.eval_build = True
        elif self.eval_build:
            self.eval_build = False
        else:
            return

        if self.Hermitian:
            self.weights = torch.cat(
                [
                    self.W["y0_modes"],
                    self.W["00_modes"].cfloat(),
                    self.W["y0_modes"].flip(dims=(-2,)).conj(),
                ],
                dim=-2,
            )
            self.weights = torch.cat([self.weights, self.W["yposx_modes"]], dim=-1)
            self.weights = torch.cat(
                [self.weights[..., 1:].conj().rot90(k=2, dims=[-2, -1]), self.weights],
                dim=-1,
            )
        else:
            self.weights = self.W[:]

        if self.first_layer or self.last_layer:

            # construct the weight
            self.weights = self.weights.repeat(1, self.group_size, 1, 1, 1)

            # apply each of the group elements to the corresponding repetition
            for k in range(1, self.rt_group_size):
                self.weights[:, k] = self.weights[:, k].rot90(k=k, dims=[-2, -1])

            # apply each the reflection group element to the rotated kernels
            if self.reflection:
                self.weights[:, self.rt_group_size :] = self.weights[
                    :, : self.rt_group_size
                ].flip(dims=[-2])

            # collapse out_channels and group1 dimensions for use with conv2d
            if self.first_layer:
                self.weights = self.weights.view(
                    -1, self.in_channels, self.kernel_size_Y, self.kernel_size_Y
                )
                if self.B is not None:
                    self.bias = self.B.repeat_interleave(repeats=self.group_size, dim=1)
            else:
                self.weights = self.weights.transpose(2, 1).reshape(
                    self.out_channels, -1, self.kernel_size_Y, self.kernel_size_Y
                )
                self.bias = self.B

        else:

            # construct the weight
            self.weights = self.weights.repeat(1, self.group_size, 1, 1, 1, 1)

            # apply elements in the rotation group
            for k in range(1, self.rt_group_size):
                self.weights[:, k] = self.weights[:, k - 1].rot90(dims=[-2, -1])

                if self.reflection:
                    self.weights[:, k] = torch.cat(
                        [
                            self.weights[:, k, :, self.rt_group_size - 1].unsqueeze(2),
                            self.weights[:, k, :, : (self.rt_group_size - 1)],
                            self.weights[:, k, :, (self.rt_group_size + 1) :],
                            self.weights[:, k, :, self.rt_group_size].unsqueeze(2),
                        ],
                        dim=2,
                    )
                else:
                    self.weights[:, k] = torch.cat(
                        [
                            self.weights[:, k, :, -1].unsqueeze(2),
                            self.weights[:, k, :, :-1],
                        ],
                        dim=2,
                    )

            if self.reflection:
                # apply elements in the reflection group
                self.weights[:, self.rt_group_size :] = torch.cat(
                    [
                        self.weights[:, : self.rt_group_size, :, self.rt_group_size :],
                        self.weights[:, : self.rt_group_size, :, : self.rt_group_size],
                    ],
                    dim=3,
                ).flip([-2])

            # collapse out_channels / groups1 and in_channels/groups2 dimensions for use with conv2d
            self.weights = self.weights.view(
                self.out_channels * self.group_size,
                self.in_channels * self.group_size,
                self.kernel_size_Y,
                self.kernel_size_Y,
            )
            if self.B is not None:
                self.bias = self.B.repeat_interleave(repeats=self.group_size, dim=1)

        if self.Hermitian:
            self.weights = self.weights[..., -self.kernel_size_X :]

    def forward(self, x):

        self.get_weight()

        # output is of shape (batch * out_channels, number of group elements, ny, nx)
        x = nn.functional.conv2d(input=x, weight=self.weights)

        # add the bias
        if self.B is not None:
            x = x + self.bias
        return x


class GSpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes, reflection=False):
        super(GSpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes = modes
        self.conv = GConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2 * modes - 1,
            reflection=reflection,
            bias=False,
            spectral=True,
            Hermitian=True,
        )
        self.get_weight()

    # Building the weight
    def get_weight(self):
        self.conv.get_weight()
        self.weights = self.conv.weights.transpose(0, 1)

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        # get the index of the zero frequency and construct weight
        freq0_y = (
            (torch.fft.fftshift(torch.fft.fftfreq(x.shape[-2])) == 0).nonzero().item()
        )
        self.get_weight()

        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.fftshift(torch.fft.rfft2(x), dim=-2)
        x_ft = x_ft[
            ..., (freq0_y - self.modes + 1) : (freq0_y + self.modes), : self.modes
        ]

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.weights.shape[0],
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[
            ..., (freq0_y - self.modes + 1) : (freq0_y + self.modes), : self.modes
        ] = self.compl_mul2d(x_ft, self.weights)

        # Return to physical space
        x = torch.fft.irfft2(
            torch.fft.ifftshift(out_ft, dim=-2), s=(x.size(-2), x.size(-1))
        )

        return x


class GMLP2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels,
        reflection=False,
        last_layer=False,
    ):
        super(GMLP2d, self).__init__()
        self.mlp1 = GConv2d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            reflection=reflection,
        )
        self.mlp2 = GConv2d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            reflection=reflection,
            last_layer=last_layer,
        )

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x


class GNorm(nn.Module):
    def __init__(self, width, group_size):
        super().__init__()
        self.group_size = group_size
        self.norm = torch.nn.InstanceNorm3d(width)

    def forward(self, x):
        x = x.view(x.shape[0], -1, self.group_size, x.shape[-2], x.shape[-1])
        x = self.norm(x)
        x = x.view(x.shape[0], -1, x.shape[-2], x.shape[-1])
        return x


class Model(nn.Module):
    def __init__(self, args, device, s1=96, s2=96):
        super(Model, self).__init__()
        self.__name__ = "GFNO"
        self.args = args
        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input shape: (batchsize, c=self.in_channels, x=64, y=64)
        output: the solution of the next timesteps
        output shape: (batchsize, c=self.out_channels, x=64, y=64)
        """
        self.in_channels = args.fun_dim
        self.out_channels = args.out_dim
        self.modes = args.modes
        self.width = args.n_hidden

        reflection = False

        if args.unified_pos and args.geotype != "unstructured":
            self.pos = unified_pos_embedding(args.shapelist, args.ref, device=device)
            in_dim = args.fun_dim + args.ref ** len(args.shapelist)
        else:
            in_dim = args.fun_dim + args.space_dim
        self.preprocess = MLP(
            in_dim,
            args.n_hidden * 2,
            args.n_hidden,
            n_layers=0,
            res=False,
            act=args.act,
        )

        # 2) 时间嵌入（可选）
        if args.time_input:
            self.time_fc = nn.Sequential(
                nn.Linear(args.n_hidden, args.n_hidden),
                nn.SiLU(),
                nn.Linear(args.n_hidden, args.n_hidden),
            )
        else:
            self.time_fc = None

        # 非结构化投影器（把点云 ↔ 规则网格）
        if args.geotype == "unstructured":
            self.group_size = 4 * (1 + reflection)
            self.fftproject_in = SpectralConv2d_IrregularGeo(
                in_channels=args.n_hidden,
                out_channels=args.n_hidden,
                modes1=args.modes,
                modes2=args.modes,
                s1=s1,
                s2=s2,
            )
            # 这里要改：out 通道 = out_dim
            self.fftproject_out = SpectralConv2d_IrregularGeo(
                in_channels=self.width * self.group_size,
                out_channels=self.width,
                modes1=args.modes,
                modes2=args.modes,
                s1=s1,
                s2=s2,
            )
            self.iphi = IPHI()
            # 规则网格大小来自 s1, s2（可用 args 里 shapelist 推断）
            grid_h, grid_w = s1, s2
            self.point_q = nn.Sequential(
                nn.Linear(self.width, self.width * 4),
                nn.GELU(),
                nn.Linear(self.width * 4, self.out_channels),
            )
        else:
            grid_h, grid_w = args.shapelist
            self.q = GMLP2d(
                in_channels=self.width,
                out_channels=self.out_channels,
                mid_channels=self.width * 4,
                reflection=reflection,
                last_layer=True,
            )

        self.padding = [(16 - size % 16) % 16 for size in [grid_h, grid_w]]

        self.p = GConv2d(
            in_channels=args.n_hidden,
            out_channels=self.width,
            kernel_size=1,
            reflection=reflection,
            first_layer=True,
        )
        self.conv0 = GSpectralConv2d(
            in_channels=self.width,
            out_channels=self.width,
            modes=self.modes,
            reflection=reflection,
        )
        self.conv1 = GSpectralConv2d(
            in_channels=self.width,
            out_channels=self.width,
            modes=self.modes,
            reflection=reflection,
        )
        self.conv2 = GSpectralConv2d(
            in_channels=self.width,
            out_channels=self.width,
            modes=self.modes,
            reflection=reflection,
        )
        self.conv3 = GSpectralConv2d(
            in_channels=self.width,
            out_channels=self.width,
            modes=self.modes,
            reflection=reflection,
        )
        self.mlp0 = GMLP2d(
            in_channels=self.width,
            out_channels=self.width,
            mid_channels=self.width,
            reflection=reflection,
        )
        self.mlp1 = GMLP2d(
            in_channels=self.width,
            out_channels=self.width,
            mid_channels=self.width,
            reflection=reflection,
        )
        self.mlp2 = GMLP2d(
            in_channels=self.width,
            out_channels=self.width,
            mid_channels=self.width,
            reflection=reflection,
        )
        self.mlp3 = GMLP2d(
            in_channels=self.width,
            out_channels=self.width,
            mid_channels=self.width,
            reflection=reflection,
        )
        self.w0 = GConv2d(
            in_channels=self.width,
            out_channels=self.width,
            kernel_size=1,
            reflection=reflection,
        )
        self.w1 = GConv2d(
            in_channels=self.width,
            out_channels=self.width,
            kernel_size=1,
            reflection=reflection,
        )
        self.w2 = GConv2d(
            in_channels=self.width,
            out_channels=self.width,
            kernel_size=1,
            reflection=reflection,
        )
        self.w3 = GConv2d(
            in_channels=self.width,
            out_channels=self.width,
            kernel_size=1,
            reflection=reflection,
        )
        self.norm = GNorm(self.width, group_size=4 * (1 + reflection))

    # ===== 两条数据路径：与 FNO 对齐 =====
    def structured_geo(self, x, fx, T=None):
        """
        x: (B, N, space_dim)  规则网格上的坐标序列（按 shapelist 拉直）
        fx: (B, N, fun_dim)   每点特征
        返回: (B, N, out_dim)
        """
        B, N, _ = x.shape

        # 位置嵌入或直接用 x
        if self.args.unified_pos:
            pos = self.pos.repeat(B, 1, 1)  # (B, N, pos_dim)
            feats = torch.cat([pos, fx], dim=-1) if fx is not None else pos
        else:
            feats = torch.cat([x, fx], dim=-1) if fx is not None else x

        # 预处理到 n_hidden
        feats = self.preprocess(feats)  # (B, N, n_hidden)

        # 加时间嵌入
        if (T is not None) and (self.time_fc is not None):
            t_emb = timestep_embedding(T, self.args.n_hidden).repeat(1, N, 1)
            feats = feats + self.time_fc(t_emb)

        # reshape -> (B, C=n_hidden, H, W)
        H, W = self.args.shapelist
        xg = feats.permute(0, 2, 1).reshape(B, self.args.n_hidden, H, W)

        # padding (右/下)
        if not all(p == 0 for p in self.padding):
            xg = F.pad(xg, [0, self.padding[1], 0, self.padding[0]])

        # === GFNO 主干 ===
        xg = self._gfno_stem_forward(xg)

        # 去 padding
        if not all(p == 0 for p in self.padding):
            xg = xg[..., : -self.padding[0], : -self.padding[1]]

        # 投到 out_dim，并拉回 (B, N, out_dim)
        xg = self.q(xg)  # (B, out_dim, H, W)
        out = xg.reshape(B, self.out_channels, -1).permute(0, 2, 1)
        return out

    def unstructured_geo(self, x, fx, T=None):
        """
        x:  (B, N, space_dim)   非规则点云坐标
        fx: (B, N, fun_dim)
        返回: (B, N, out_dim)
        """
        B, N, _ = x.shape

        feats = torch.cat([x, fx], dim=-1) if fx is not None else x
        feats = self.preprocess(feats)  # (B, N, n_hidden)

        if (T is not None) and (self.time_fc is not None):
            t_emb = timestep_embedding(T, self.args.n_hidden).repeat(1, N, 1)
            feats = feats + self.time_fc(t_emb)

        # 投影到规则网格: (B, C, H, W)
        xg = self.fftproject_in(
            feats.permute(0, 2, 1), x_in=x, iphi=self.iphi, code=None
        )

        # === GFNO 主干 ===
        xg = self._gfno_stem_forward(xg)

        # # 先把通道降到 out_dim=1
        # xg = self.q(xg)  # (B, out_dim, H, W)

        # project grid -> points *before* head, staying in width channels
        xp = self.fftproject_out(xg, x_out=x, iphi=self.iphi, code=None).permute(
            0, 2, 1
        )  # (B, N, width)

        # pointwise head to out_dim
        out = self.point_q(xp)  # (B, N, out_dim)

        return out

    def _gfno_stem_forward(self, x):
        """
        x: (B, C, H, W)
        """
        # 首层抬升
        x = self.p(x)

        # 4 层 (Norm -> Spectral -> Norm -> GMLP) + 1x1 残差
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

    # 与 FNO 对齐的 forward 签名（保留兼容性）
    def forward(self, x, fx=None, T=None, geo=None):
        """
        与 FNO 一致：
          - structured: x=(B,N,space_dim), fx=(B,N,fun_dim)
          - unstructured: 同上；内部会用投影器做网格 ↔ 点云变换
        输出: (B, N, out_dim)
        """
        if self.args.geotype == "unstructured":
            return self.unstructured_geo(x, fx, T)
        else:
            return self.structured_geo(x, fx, T)
