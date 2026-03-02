import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from onescience.modules import OneEncoder, OneDecoder, OneFuser

from onescience.models.meta import ModelMetaData


@dataclass
class MetaData(ModelMetaData):
    name: str = "Fengwu"
    # Optimization
    jit: bool = False  # ONNX Ops Conflict
    cuda_graphs: bool = True
    amp: bool = True
    # Inference
    onnx_cpu: bool = False  # No FFT op on CPU
    onnx_gpu: bool = True
    onnx_runtime: bool = True
    # Physics informed
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False


class Fengwu(nn.Module):
    """
    FengWu PyTorch impl of: `FengWu: Pushing the Skillful Global Medium-range Weather Forecast beyond 10 Days Lead`
    - https://arxiv.org/pdf/2304.02948.pdf

    Args:
        img_size: Image size(Lat, Lon). Default: (721,1440)
        pressure_level: Number of pressure_level. Default: 37
        embed_dim (int): Patch embedding dimension. Default: 192
        patch_size (tuple[int]): Patch token size. Default: (4,4)
        num_heads (tuple[int]): Number of attention heads in different layers.
        window_size (tuple[int]): Window size.
    """

    def __init__(
        self,
        img_size=(721, 1440),
        pressure_level=37,
        embed_dim=192,
        patch_size=(4, 4),
        num_heads=(6, 12, 12, 6),
        window_size=(2, 6, 12),
    ):
        super().__init__()
        resolution_down1 = (
            math.ceil(img_size[0] / patch_size[0]),
            math.ceil(img_size[1] / patch_size[1]),
        )
        resolution_down2 = (
            math.ceil(resolution_down1[0] / 2),
            math.ceil(resolution_down1[1] / 2),
        )
        resolution = (resolution_down1, resolution_down2)
        self.encoder_surface = OneEncoder(
            style="FengWuEncoder",
            in_chans=4,
            input_resolution=resolution[0],
            middle_resolution=resolution[1],
        )

        self.encoder_z = OneEncoder(
            style="FengWuEncoder",
            input_resolution=resolution[0],
            middle_resolution=resolution[1],
        )

        self.encoder_r = OneEncoder(
            style="FengWuEncoder",
            input_resolution=resolution[0],
            middle_resolution=resolution[1],
        )

        self.encoder_u = OneEncoder(
            style="FengWuEncoder",
            in_chans=pressure_level,
            input_resolution=resolution[0],
            middle_resolution=resolution[1],
        )

        self.encoder_v = OneEncoder(
            style="FengWuEncoder",
            input_resolution=resolution[0],
            middle_resolution=resolution[1],
        )

        self.encoder_t = OneEncoder(
            style="FengWuEncoder",
            input_resolution=resolution[0],
            middle_resolution=resolution[1],
        )

        self.fuser = OneFuser(
            style="FengWuFuser",
            input_resolution=(6, resolution[1][0], resolution[1][1])
        )

        self.decoder_surface = OneDecoder(
            style="FengWuDecoder",
            input_resolution=resolution[0],
            output_resolution=resolution[1],
            out_chans=4
        )

        self.decoder_z = OneDecoder(
            style="FengWuDecoder",
            input_resolution=resolution[0],
            output_resolution=resolution[1]
        )

        self.decoder_r = OneDecoder(
            style="FengWuDecoder",
            input_resolution=resolution[0],
            output_resolution=resolution[1]
        )

        self.decoder_u =OneDecoder(
            style="FengWuDecoder",
            input_resolution=resolution[0],
            output_resolution=resolution[1]
        )

        self.decoder_v = OneDecoder(
            style="FengWuDecoder",
            input_resolution=resolution[0],
            output_resolution=resolution[1]
        )

        self.decoder_t = OneDecoder(
            style="FengWuDecoder",
            input_resolution=resolution[0],
            output_resolution=resolution[1]
        )

    def forward(self, surface, z, r, u, v, t):
        """
        Args:
            surface (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=4.
            z (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=37.
            r (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=37.
            u (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=37.
            v (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=37.
            t (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=37.
        """

        surface, skip_surface = self.encoder_surface(surface)
        z, skip_z = self.encoder_z(z)
        r, skip_r = self.encoder_r(r)
        u, skip_u = self.encoder_u(u)
        v, skip_v = self.encoder_v(v)
        t, skip_t = self.encoder_t(t)

        x = torch.concat(
            [
                surface.unsqueeze(1),
                z.unsqueeze(1),
                r.unsqueeze(1),
                u.unsqueeze(1),
                v.unsqueeze(1),
                t.unsqueeze(1),
            ],
            dim=1,
        )
        B, PL, L_SIZE, C = x.shape
        x = x.reshape(B, -1, C)
        x = self.fuser(x)

        x = x.reshape(B, PL, L_SIZE, C)
        surface, z, r, u, v, t = (
            x[:, 0, :, :],
            x[:, 1, :, :],
            x[:, 2, :, :],
            x[:, 3, :, :],
            x[:, 4, :, :],
            x[:, 5, :, :],
        )

        surface = self.decoder_surface(surface, skip_surface)
        z = self.decoder_z(z, skip_z)
        r = self.decoder_r(r, skip_r)
        u = self.decoder_u(u, skip_u)
        v = self.decoder_v(v, skip_v)
        t = self.decoder_t(t, skip_t)
        return surface, z, r, u, v, t
