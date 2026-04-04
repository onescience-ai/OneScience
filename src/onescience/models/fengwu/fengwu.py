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
    FengWu 的主模型实现。

    该模型由多个变量分支编码器、一个中分辨率三维 fuser，以及多个变量分支解码器组成。

    结构顺序为：

    - 多个 `OneEncoder(style="FengWuEncoder")`
      - 分别编码 surface、Z、R、U、V、T 六个变量分支
    - `OneFuser(style="FengWuFuser")`
      - 在统一三维网格 `(Variables, Height, Width)` 上融合中分辨率特征
    - 多个 `OneDecoder(style="FengWuDecoder")`
      - 分别恢复各变量分支输出

    与 Pangu 不同，FengWu 不把所有变量直接拼成单一路径输入，而是先按变量族分支编码，
    再在中分辨率层面做跨变量三维融合。

    Reference:
    - `FengWu: Pushing the Skillful Global Medium-range Weather Forecast beyond 10 Days Lead`
    - https://arxiv.org/pdf/2304.02948.pdf

    Args:
        img_size (tuple[int, int]):
            输入场空间尺寸 `(Height, Width)`。
        pressure_level (int):
            高空变量的层数。
        embed_dim (int):
            编码器高分辨率特征维度。
        patch_size (tuple[int, int]):
            二维 patch 切分尺寸 `(PatchHeight, PatchWidth)`。
        num_heads (tuple[int, int, int, int]):
            注意力头数配置，约定顺序为：
            `(EncoderHighHeads, EncoderMiddleHeads, DecoderMiddleHeads, DecoderHighHeads)`。
        window_size (tuple[int, int, int]):
            FengWuFuser 的三维窗口大小。
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
        input_resolution = (
            math.ceil(img_size[0] / patch_size[0]),
            math.ceil(img_size[1] / patch_size[1]),
        )
        middle_resolution = (
            math.ceil(input_resolution[0] / 2),
            math.ceil(input_resolution[1] / 2),
        )
        encoder_num_heads = (num_heads[0], num_heads[1])
        decoder_num_heads = (num_heads[3], num_heads[2])
        drop_path = np.linspace(0, 0.2, 8).tolist()

        self.encoder_surface = OneEncoder(
            style="FengWuEncoder",
            input_resolution=input_resolution,
            middle_resolution=middle_resolution,
            in_chans=4,
            img_size=img_size,
            patch_size=patch_size,
            dim=embed_dim,
            num_heads=encoder_num_heads,
            window_size=window_size[1:],
            drop_path=drop_path,
        )
        self.encoder_z = OneEncoder(
            style="FengWuEncoder",
            input_resolution=input_resolution,
            middle_resolution=middle_resolution,
            in_chans=pressure_level,
            img_size=img_size,
            patch_size=patch_size,
            dim=embed_dim,
            num_heads=encoder_num_heads,
            window_size=window_size[1:],
            drop_path=drop_path,
        )
        self.encoder_r = OneEncoder(
            style="FengWuEncoder",
            input_resolution=input_resolution,
            middle_resolution=middle_resolution,
            in_chans=pressure_level,
            img_size=img_size,
            patch_size=patch_size,
            dim=embed_dim,
            num_heads=encoder_num_heads,
            window_size=window_size[1:],
            drop_path=drop_path,
        )
        self.encoder_u = OneEncoder(
            style="FengWuEncoder",
            input_resolution=input_resolution,
            middle_resolution=middle_resolution,
            in_chans=pressure_level,
            img_size=img_size,
            patch_size=patch_size,
            dim=embed_dim,
            num_heads=encoder_num_heads,
            window_size=window_size[1:],
            drop_path=drop_path,
        )
        self.encoder_v = OneEncoder(
            style="FengWuEncoder",
            input_resolution=input_resolution,
            middle_resolution=middle_resolution,
            in_chans=pressure_level,
            img_size=img_size,
            patch_size=patch_size,
            dim=embed_dim,
            num_heads=encoder_num_heads,
            window_size=window_size[1:],
            drop_path=drop_path,
        )
        self.encoder_t = OneEncoder(
            style="FengWuEncoder",
            input_resolution=input_resolution,
            middle_resolution=middle_resolution,
            in_chans=pressure_level,
            img_size=img_size,
            patch_size=patch_size,
            dim=embed_dim,
            num_heads=encoder_num_heads,
            window_size=window_size[1:],
            drop_path=drop_path,
        )

        self.fuser = OneFuser(
            style="FengWuFuser",
            input_resolution=(6, *middle_resolution),
            dim=embed_dim * 2,
            num_heads=num_heads[2],
            window_size=window_size,
            drop_path=drop_path[2:],
        )

        self.decoder_surface = OneDecoder(
            style="FengWuDecoder",
            output_resolution=input_resolution,
            middle_resolution=middle_resolution,
            out_chans=4,
            img_size=img_size,
            patch_size=patch_size,
            dim=embed_dim,
            num_heads=decoder_num_heads,
            window_size=window_size[1:],
            drop_path=drop_path,
        )
        self.decoder_z = OneDecoder(
            style="FengWuDecoder",
            output_resolution=input_resolution,
            middle_resolution=middle_resolution,
            out_chans=pressure_level,
            img_size=img_size,
            patch_size=patch_size,
            dim=embed_dim,
            num_heads=decoder_num_heads,
            window_size=window_size[1:],
            drop_path=drop_path,
        )
        self.decoder_r = OneDecoder(
            style="FengWuDecoder",
            output_resolution=input_resolution,
            middle_resolution=middle_resolution,
            out_chans=pressure_level,
            img_size=img_size,
            patch_size=patch_size,
            dim=embed_dim,
            num_heads=decoder_num_heads,
            window_size=window_size[1:],
            drop_path=drop_path,
        )
        self.decoder_u = OneDecoder(
            style="FengWuDecoder",
            output_resolution=input_resolution,
            middle_resolution=middle_resolution,
            out_chans=pressure_level,
            img_size=img_size,
            patch_size=patch_size,
            dim=embed_dim,
            num_heads=decoder_num_heads,
            window_size=window_size[1:],
            drop_path=drop_path,
        )
        self.decoder_v = OneDecoder(
            style="FengWuDecoder",
            output_resolution=input_resolution,
            middle_resolution=middle_resolution,
            out_chans=pressure_level,
            img_size=img_size,
            patch_size=patch_size,
            dim=embed_dim,
            num_heads=decoder_num_heads,
            window_size=window_size[1:],
            drop_path=drop_path,
        )
        self.decoder_t = OneDecoder(
            style="FengWuDecoder",
            output_resolution=input_resolution,
            middle_resolution=middle_resolution,
            out_chans=pressure_level,
            img_size=img_size,
            patch_size=patch_size,
            dim=embed_dim,
            num_heads=decoder_num_heads,
            window_size=window_size[1:],
            drop_path=drop_path,
        )

        self.img_size = img_size
        self.pressure_level = pressure_level
        self.patch_size = patch_size
        self.input_resolution = input_resolution
        self.middle_resolution = middle_resolution

    def forward(self, surface, z, r, u, v, t):
        """
        Args:
            surface (torch.Tensor):
                `(Batch, 4, Height, Width)`。
            z (torch.Tensor):
                `(Batch, pressure_level, Height, Width)`。
            r (torch.Tensor):
                `(Batch, pressure_level, Height, Width)`。
            u (torch.Tensor):
                `(Batch, pressure_level, Height, Width)`。
            v (torch.Tensor):
                `(Batch, pressure_level, Height, Width)`。
            t (torch.Tensor):
                `(Batch, pressure_level, Height, Width)`。

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                六个变量分支的输出，shape 与各自输入分支一致。
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
        Batch, Variables, NumTokensPerVariable, Channels = x.shape
        x = x.reshape(Batch, -1, Channels)
        x = self.fuser(x)

        x = x.reshape(Batch, Variables, NumTokensPerVariable, Channels)
        surface, z, r, u, v, t = (
            x[:, 0, :, :],
            x[:, 1, :, :],
            x[:, 2, :, :],
            x[:, 3, :, :],
            x[:, 4, :, :],
            x[:, 5, :, :],
        )

        surface = self.decoder_surface([surface, skip_surface])
        z = self.decoder_z([z, skip_z])
        r = self.decoder_r([r, skip_r])
        u = self.decoder_u([u, skip_u])
        v = self.decoder_v([v, skip_v])
        t = self.decoder_t([t, skip_t])
        return surface, z, r, u, v, t
