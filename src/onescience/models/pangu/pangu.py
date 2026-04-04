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
    Pangu-Weather 的主模型实现。

    该模型使用以下组件完成编码、主干特征提取与输出恢复：

    - `OneEmbedding(style="PanguEmbedding")`
      - 将 surface 分支与 upper-air 分支分别映射到 patch 特征空间
    - `OneFuser(style="PanguFuser")`
      - 在统一的三维 token 网格上完成主干特征融合
    - `OneSample(style="PanguDownSample" / "PanguUpSample")`
      - 在主干中完成空间尺度变换
    - `OneRecovery(style="PanguPatchRecovery")`
      - 将 patch 级特征恢复为目标物理场

    在当前实现中：

    - 输入包含 7 个 surface 通道，其中 4 个是待预测地表变量，3 个是静态掩码
    - 输入包含 5 个 upper-air 变量，每个变量对应 13 个气压层
    - surface patch token 会先在 `PressureLevels` 维补成长度为 1
    - 然后与 upper-air patch token 沿 `PressureLevels` 维拼接
    - 之后统一送入三维 `PanguFuser` 主干

    Reference:
    - `Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Global Weather Forecast`
    - https://arxiv.org/abs/2211.02556

    Args:
        img_size (tuple[int, int]):
            输入空间尺寸 `(Height, Width)`。
        patch_size (tuple[int, int, int]):
            patch 切分尺寸 `(PatchPressureLevels, PatchHeight, PatchWidth)`。
        embed_dim (int):
            patch embedding 后的特征维度。
        num_heads (tuple[int, int, int, int]):
            四个主干阶段对应的注意力头数。
        window_size (tuple[int, int, int]):
            三维窗口大小 `(PressureLevelsWindow, HeightWindow, WidthWindow)`。
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
        # Surface input contains 4 predicted variables and 3 static masks:
        # topography mask, land-sea mask, and soil type mask.

        self.patchembed2d = OneEmbedding(
            style="PanguEmbedding",
            img_size=img_size,
            patch_size=patch_size[1:],
            Variables=7,
            embed_dim=embed_dim,
        )
        self.patchembed3d = OneEmbedding(
            style="PanguEmbedding",
            img_size=(13, *img_size),
            patch_size=patch_size,
            Variables=5,
            embed_dim=embed_dim,
        )

        patched_input_shape = (
            8,
            math.ceil(img_size[0] / patch_size[1]),
            math.ceil(img_size[1] / patch_size[2]),
        )

        self.layer1 = OneFuser(
            style="PanguFuser",
            dim=embed_dim,
            input_resolution=patched_input_shape,
            depth=2,
            num_heads=num_heads[0],
            window_size=window_size,
            drop_path=drop_path[:2],
        )

        patched_downsampled_shape = (
            8,
            math.ceil(patched_input_shape[1] / 2),
            math.ceil(patched_input_shape[2] / 2),
        )

        self.downsample = OneSample(
            style="PanguDownSample",
            in_dim=embed_dim,
            input_resolution=patched_input_shape,
            output_resolution=patched_downsampled_shape,
        )
        self.layer2 = OneFuser(
            style="PanguFuser",
            dim=embed_dim * 2,
            input_resolution=patched_downsampled_shape,
            depth=6,
            num_heads=num_heads[1],
            window_size=window_size,
            drop_path=drop_path[2:],
        )
        self.layer3 = OneFuser(
            style="PanguFuser",
            dim=embed_dim * 2,
            input_resolution=patched_downsampled_shape,
            depth=6,
            num_heads=num_heads[2],
            window_size=window_size,
            drop_path=drop_path[2:],
        )
        self.upsample = OneSample(
            style="PanguUpSample",
            in_dim=embed_dim * 2,
            out_dim=embed_dim,
            input_resolution=patched_downsampled_shape,
            output_resolution=patched_input_shape,
        )
        self.layer4 = OneFuser(
            style="PanguFuser",
            dim=embed_dim,
            input_resolution=patched_input_shape,
            depth=2,
            num_heads=num_heads[3],
            window_size=window_size,
            drop_path=drop_path[:2],
        )

        # The recovered surface output contains only the 4 prognostic surface variables.
        # Static masks are input-only features and are not part of the prediction target.
        self.patchrecovery2d = OneRecovery(
            style="PanguPatchRecovery",
            img_size=(721, 1440),
            patch_size=(4, 4),
            in_chans=embed_dim * 2,
            out_chans=4,
        )
        self.patchrecovery3d = OneRecovery(
            style="PanguPatchRecovery",
            img_size=(13, 721, 1440),
            patch_size=(2, 4, 4),
            in_chans=embed_dim * 2,
            out_chans=5,
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor):
                Input tensor with shape `(Batch, 4 + 3 + 5 * 13, Height, Width)`.

                Channel layout:
                - first 4 channels: prognostic surface variables
                - next 3 channels: static masks
                - remaining `5 * 13` channels: upper-air variables flattened over pressure levels

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - surface output:
                  `(Batch, 4, Height, Width)`
                - upper-air output:
                  `(Batch, 5, 13, Height, Width)`
        """
        SurfaceInput = x[:, :7, :, :]
        UpperAirInput = x[:, 7:, :, :].reshape(x.shape[0], 5, 13, x.shape[2], x.shape[3])

        SurfaceFeatures = self.patchembed2d(SurfaceInput)
        UpperAirFeatures = self.patchembed3d(UpperAirInput)

        CombinedFeatures = torch.concat(
            [SurfaceFeatures.unsqueeze(2), UpperAirFeatures], dim=2
        )
        Batch, Channels, PressureLevels, Height, Width = CombinedFeatures.shape
        Tokens = CombinedFeatures.reshape(Batch, Channels, -1).transpose(1, 2)

        Tokens = self.layer1(Tokens)
        SkipTokens = Tokens

        Tokens = self.downsample(Tokens)
        Tokens = self.layer2(Tokens)
        Tokens = self.layer3(Tokens)
        Tokens = self.upsample(Tokens)
        Tokens = self.layer4(Tokens)

        OutputFeatures = torch.concat([Tokens, SkipTokens], dim=-1)
        OutputFeatures = OutputFeatures.transpose(1, 2).reshape(
            Batch, -1, PressureLevels, Height, Width
        )
        output_surface = OutputFeatures[:, :, 0, :, :]
        output_upper_air = OutputFeatures[:, :, 1:, :, :]

        output_surface = self.patchrecovery2d(output_surface)
        output_upper_air = self.patchrecovery3d(output_upper_air)
        return output_surface, output_upper_air
