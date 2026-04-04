import torch
from torch import nn


class PanguPatchRecovery(nn.Module):
    """
        Pangu-Weather模型的统一Patch Recovery模块。

        Pangu-Weather的输出特征解码阶段，负责将patch级别的特征表示恢复为原始气象场分辨率，
        将高维特征图映射回具有实际物理含义的二维或三维变量场。

        输入支持二维和三维输入，会被拆成两条分支：
        - 二维输入（例如地表变量分支）形状为：
          (Batch, Channels, Height, Width)
        - 三维输入（例如大气变量分支）形状为：
          (Batch, Channels, Pressure Levels, Height, Width)

        实现逻辑统一使用三维patch recovery逻辑：
        - 先根据patch_size使用ConvTranspose3d对特征图进行反卷积恢复；
        - 再按照img_size对恢复后的结果进行中心裁剪，使输出空间尺寸与目标场对齐；
        - 若输入是二维张量，则先在Pressure Levels位置补一个长度为1的伪三维维度，完成三维恢复后再将该维度去掉。

        因此，该模块同时支持二维输入和三维输入，但内部始终走统一的三维实现。

        Args:
            img_size (tuple[int, int] | tuple[int, int, int]):
                输出场空间尺寸。
                - 二维输入对应 (Height, Width)
                - 三维输入对应 (Pressure Levels, Height, Width)
            patch_size (tuple[int, int] | tuple[int, int, int]):
                patch 的恢复尺寸。
                - 二维输入对应 (Patch Height, Patch Width)
                - 三维输入对应 (Patch Pressure Levels, Patch Height, Patch Width)
            in_chans (int):
                输入特征通道数。
            out_chans (int):
                输出变量通道数。

        形状:
            输入:
                - 二维输入:
                  [Batch, in_chans, Height, Width]
                - 三维输入:
                  [Batch, in_chans, Pressure Levels, Height, Width]
            输出:
                - 二维输入对应输出:
                  [Batch, out_chans, Out Height, Out Width]
                - 三维输入对应输出:
                  [Batch, out_chans, Out Pressure Levels, Out Height, Out Width]

            各维含义与常见取值：
                - Batch：批大小，即一次前向传播中的样本数，例如 1、2、4、8。
                - in_chans：输入特征图通道数，常见为 384。
                - out_chans：输出变量数。
                - Pressure Levels：气压层数。
                - Height：patch级特征图的纬向网格数量，例如 181。
                - Width：patch级特征图的经向网格数量，例如 360。
                - Out Pressure Levels：恢复后的目标气压层数，例如 13。
                - Out Height：恢复后的目标纬向网格数量，例如 721。
                - Out Width：恢复后的目标经向网格数量，例如 1440。

        Example:
            >>> # Pangu-Weather 中的 surface 分支
            >>> Batch = 2
            >>> img_size = (721, 1440)
            >>> patch_size = (4, 4)
            >>> in_chans = 384
            >>> out_chans = 7
            >>> surface_patch_recovery = OneRecovery(
            ...     style="PanguPatchRecovery",
            ...     img_size=img_size,
            ...     patch_size=patch_size,
            ...     in_chans=in_chans,
            ...     out_chans=out_chans
            ... ).cuda()
            >>> surface_x = torch.randn(Batch, in_chans, 181, 360).cuda()
            >>> surface_out = surface_patch_recovery(surface_x)
            >>> surface_out.shape
            torch.Size([2, 7, 721, 1440])

            >>> # Pangu-Weather 中的 upper-air 分支
            >>> Batch = 2
            >>> img_size = (13, 721, 1440)
            >>> patch_size = (2, 4, 4)
            >>> in_chans = 384
            >>> out_chans = 5
            >>> upper_air_patch_recovery = OneRecovery(
            ...     style="PanguPatchRecovery",
            ...     img_size=img_size,
            ...     patch_size=patch_size,
            ...     in_chans=in_chans,
            ...     out_chans=out_chans
            ... ).cuda()
            >>> upper_air_x = torch.randn(Batch, in_chans, 7, 181, 360).cuda()
            >>> upper_air_out = upper_air_patch_recovery(upper_air_x)
            >>> upper_air_out.shape
            torch.Size([2, 5, 13, 721, 1440])
    """

    def __init__(
        self,
        img_size=(13, 721, 1440),
        patch_size=(2, 4, 4),
        in_chans=192 * 2,
        out_chans=5,
    ):
        super().__init__()

        if len(img_size) == 2:
            img_size = (1, *img_size)
        elif len(img_size) != 3:
            raise ValueError("img_size must have 2 or 3 dimensions")

        if len(patch_size) == 2:
            patch_size = (1, *patch_size)
        elif len(patch_size) != 3:
            raise ValueError("patch_size must have 2 or 3 dimensions")

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.proj = nn.ConvTranspose3d(
            in_chans,
            out_chans,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor):
        squeeze_level_dim = False
        if x.ndim == 4:
            x = x.unsqueeze(2)
            squeeze_level_dim = True
        elif x.ndim != 5:
            raise ValueError("Input tensor must be 4D or 5D")

        if x.shape[1] != self.in_chans:
            raise ValueError(f"Expected input channels {self.in_chans}, but received {x.shape[1]}")

        output = self.proj(x)
        _, _, levels, height, width = output.shape

        level_pad = levels - self.img_size[0]
        height_pad = height - self.img_size[1]
        width_pad = width - self.img_size[2]

        if level_pad < 0 or height_pad < 0 or width_pad < 0:
            raise ValueError("Recovered feature map is smaller than the target img_size")

        padding_front = level_pad // 2
        padding_back = level_pad - padding_front
        padding_top = height_pad // 2
        padding_bottom = height_pad - padding_top
        padding_left = width_pad // 2
        padding_right = width_pad - padding_left

        output = output[
            :,
            :,
            padding_front : levels - padding_back,
            padding_top : height - padding_bottom,
            padding_left : width - padding_right,
        ]

        if squeeze_level_dim:
            output = output.squeeze(2)
        return output
