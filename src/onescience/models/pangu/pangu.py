import math
import torch
import numpy as np

from torch import nn
from dataclasses import dataclass
from onescience.models.meta import ModelMetaData

from onescience.modules import (
    OneEmbedding,
    OneFuser,
    OneRecovery,
    OneSample,
)

@dataclass
class MetaData(ModelMetaData):
    name: str = "Pangu"
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


class Pangu(nn.Module):
    """
    Pangu A PyTorch impl of: `Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Global Weather Forecast`
    - https://arxiv.org/abs/2211.02556

    Args:
        img_size (tuple[int]): Image size [Lat, Lon].
        patch_size (tuple[int]): Patch token size [Lat, Lon].
        embed_dim (int): Patch embedding dimension. Default: 192
        num_heads (tuple[int]): Number of attention heads in different layers.
        window_size (tuple[int]): Window size.
    """

    def __init__(
        self,
        img_size=(721, 1440),
        patch_size=(2, 4, 4),
        embed_dim=192,
        num_heads=(6, 12, 12, 6),
        window_size=(2, 6, 12),
    ):
        super().__init__()
        drop_path = np.linspace(0, 0.2, 8).tolist()
        # In addition, three constant masks(the topography mask, land-sea mask and soil type mask)
        
        self.patchembed2d = OneEmbedding(
            style="PanguEmbedding2D",
        )

        self.patchembed3d = OneEmbedding(
            style="PanguEmbedding3D",
        )
        
        patched_inp_shape = (
            8,
            math.ceil(img_size[0] / patch_size[1]),
            math.ceil(img_size[1] / patch_size[2]),
        )

        self.layer1 = OneFuser(
            style="PanGuFuser",
            dim=embed_dim,
            input_resolution=patched_inp_shape,
            depth=2,
            num_heads=num_heads[0],
            window_size=window_size,
            drop_path=drop_path[:2],
        )

        patched_inp_shape_downsample = (
            8,
            math.ceil(patched_inp_shape[1] / 2),
            math.ceil(patched_inp_shape[2] / 2),
        )
        
        self.downsample = OneSample(
            style="PanGuDownSample3D",
            in_dim=embed_dim,
            input_resolution=patched_inp_shape,
            output_resolution=patched_inp_shape_downsample,
        )
        self.layer2 = OneFuser(
            style="PanGuFuser",
            dim=embed_dim * 2,
            input_resolution=patched_inp_shape_downsample,
            depth=6,
            num_heads=num_heads[1],
            window_size=window_size,
            drop_path=drop_path[2:],
        )
        self.layer3 = OneFuser(
            style="PanGuFuser",
            dim=embed_dim * 2,
            input_resolution=patched_inp_shape_downsample,
            depth=6,
            num_heads=num_heads[2],
            window_size=window_size,
            drop_path=drop_path[2:],
        )
        self.upsample = OneSample(
            style="PanGuUpSample3D",
            in_dim=embed_dim * 2,
            out_dim=embed_dim,
            input_resolution=patched_inp_shape_downsample,
            output_resolution=patched_inp_shape
        )
        self.layer4 = OneFuser(
            style="PanGuFuser",
            dim=embed_dim,
            input_resolution=patched_inp_shape,
            depth=2,
            num_heads=num_heads[3],
            window_size=window_size,
            drop_path=drop_path[:2],
        )
        # The outputs of the 2nd encoder layer and the 7th decoder layer are concatenated along the channel dimension.
        self.patchrecovery2d = OneRecovery(
            style="pangupatchrecovery2d"
        )
        self.patchrecovery3d = OneRecovery(
            style="pangupatchrecovery3d"
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): [batch, 4+3+5*13, lat, lon]
        """
        surface = x[:, :7, :, :] # 1, 72, 721, 1440
        upper_air = x[:, 7:, :, :].reshape(x.shape[0], 5, 13, x.shape[2], x.shape[3]) # torch.Size([1, 5, 13, 721, 1440])
        surface = self.patchembed2d(surface) # torch.Size([1, 192, 181, 360])
        upper_air = self.patchembed3d(upper_air) #torch.Size([1, 192, 7, 181, 360])

        x = torch.concat([surface.unsqueeze(2), upper_air], dim=2) # torch.Size([1, 192, 8, 181, 360])
        B, C, Pl, Lat, Lon = x.shape
        x = x.reshape(B, C, -1).transpose(1, 2) # torch.Size([1, 521280, 192])
        
        x = self.layer1(x)

        skip = x

        x = self.downsample(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.upsample(x)
        x = self.layer4(x)

        output = torch.concat([x, skip], dim=-1)
        output = output.transpose(1, 2).reshape(B, -1, Pl, Lat, Lon)
        output_surface = output[:, :, 0, :, :]
        output_upper_air = output[:, :, 1:, :, :]

        output_surface = self.patchrecovery2d(output_surface)
        output_upper_air = self.patchrecovery3d(output_upper_air)
        return output_surface, output_upper_air
