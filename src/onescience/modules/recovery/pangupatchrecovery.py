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
          (Batch, Channels, PressureLevels, Height, Width)

        实现逻辑统一使用三维patch recovery逻辑：
        - 先根据patch_size使用ConvTranspose3d对特征图进行反卷积恢复；
        - 再按照img_size对恢复后的结果进行中心裁剪，使输出空间尺寸与目标场对齐；
        - 若输入是二维张量，则先在PressureLevels位置补一个长度为1的伪三维维度，完成三维恢复后再将该维度去掉。

        因此，该模块同时支持二维输入和三维输入，但内部始终走统一的三维实现。

        Args:
            img_size (tuple[int, int] | tuple[int, int, int]):
                输出场空间尺寸。
                - 二维输入对应 (Height, Width)
                - 三维输入对应 (PressureLevels, Height, Width)
            patch_size (tuple[int, int] | tuple[int, int, int]):
                patch 的恢复尺寸。
                - 二维输入对应 (PatchHeight, PatchWidth)
                - 三维输入对应 (PatchPressureLevels, PatchHeight, PatchWidth)
            in_chans (int):
                输入特征通道数。
            out_chans (int):
                输出变量通道数。

        形状:
            输入:
                - 二维输入:
                  [Batch, in_chans, Height, Width]
                - 三维输入:
                  [Batch, in_chans, PressureLevels, Height, Width]
            输出:
                - 二维输入对应输出:
                  [Batch, out_chans, OutHeight, OutWidth]
                - 三维输入对应输出:
                  [Batch, out_chans, OutPressureLevels, OutHeight, OutWidth]

            各维含义与常见取值：
                - Batch：批大小，即一次前向传播中的样本数，例如 1、2、4、8。
                - in_chans：输入特征图通道数，常见为 384。
                - out_chans：输出变量数。
                - PressureLevels：气压层数。
                - Height：patch级特征图的纬向网格数量，例如 181。
                - Width：patch级特征图的经向网格数量，例如 360。
                - OutPressureLevels：恢复后的目标气压层数，例如 13。
                - OutHeight：恢复后的目标纬向网格数量，例如 721。
                - OutWidth：恢复后的目标经向网格数量，例如 1440。

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
        SqueezePressureLevelsDim = False
        if x.ndim == 4:
            x = x.unsqueeze(2)
            SqueezePressureLevelsDim = True
        elif x.ndim != 5:
            raise ValueError("Input tensor must be 4D or 5D")

        if x.shape[1] != self.in_chans:
            raise ValueError(f"Expected input channels {self.in_chans}, but received {x.shape[1]}")

        output = self.proj(x)
        _, _, PressureLevels, Height, Width = output.shape

        PressureLevelsPad = PressureLevels - self.img_size[0]
        HeightPad = Height - self.img_size[1]
        WidthPad = Width - self.img_size[2]

        if PressureLevelsPad < 0 or HeightPad < 0 or WidthPad < 0:
            raise ValueError("Recovered feature map is smaller than the target img_size")

        PaddingFront = PressureLevelsPad // 2
        PaddingBack = PressureLevelsPad - PaddingFront
        PaddingTop = HeightPad // 2
        PaddingBottom = HeightPad - PaddingTop
        PaddingLeft = WidthPad // 2
        PaddingRight = WidthPad - PaddingLeft

        output = output[
            :,
            :,
            PaddingFront : PressureLevels - PaddingBack,
            PaddingTop : Height - PaddingBottom,
            PaddingLeft : Width - PaddingRight,
        ]

        if SqueezePressureLevelsDim:
            output = output.squeeze(2)
        return output
