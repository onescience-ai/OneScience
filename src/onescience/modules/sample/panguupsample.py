import torch
from torch import nn


class PanguUpSample(nn.Module):
    """
        Pangu-Weather模型的统一Up Sample模块。

        Pangu-Weather在多尺度特征解码阶段，使用该模块对token表示进行上采样，
        将低分辨率特征恢复到更高分辨率的特征图中，为后续Transformer/Fuser模块
        提供更精细的空间结构信息。

        输入支持二维和三维输入，会被拆成两条分支：
        - 二维输入（例如地表变量分支）对应的token形状为：
          [Batch, Height * Width, in_dim]
        - 三维输入（例如大气变量分支）对应的token形状为：
          [Batch, PressureLevels * Height * Width, in_dim]

        实现逻辑统一使用三维up sample逻辑：
        - 先根据output_resolution与input_resolution的关系，确定目标输出空间尺寸；
        - 再通过线性层将每个token的特征维扩展为适合2×2空间重排的形式；
        - 对二维输入，会先在PressureLevels位置补一个长度为1的伪三维维度；
        - 之后仅在Height和Width方向执行2×2子像素重排；
        - 最后按照output_resolution对空间尺寸进行中心裁剪，并通过LayerNorm与Linear完成通道映射。

        因此，该模块同时支持二维输入和三维输入，但内部始终走统一的三维实现。

        Args:
            input_resolution (tuple[int, int] | tuple[int, int, int]):
                输入特征图空间尺寸。
                - 二维输入对应 (Height, Width)
                - 三维输入对应 (PressureLevels, Height, Width)
            output_resolution (tuple[int, int] | tuple[int, int, int]):
                输出特征图空间尺寸。
                - 二维输入对应 (OutHeight, OutWidth)
                - 三维输入对应 (OutPressureLevels, OutHeight, OutWidth)
                其中Pangu-Weather默认只对Height和Width做2倍上采样，
                PressureLevels维度可保持不变或按目标尺寸截取。
            in_dim (int):
                输入token的特征维度。
            out_dim (int):
                输出token的特征维度。

        形状:
            输入:
                - 二维输入:
                  [Batch, Height * Width, in_dim]
                - 三维输入:
                  [Batch, PressureLevels * Height * Width, in_dim]
            输出:
                - 二维输入对应输出:
                  [Batch, OutHeight * OutWidth, out_dim]
                - 三维输入对应输出:
                  [Batch, OutPressureLevels * OutHeight * OutWidth, out_dim]

            各维含义与常见取值：
                - Batch：批大小，即一次前向传播中的样本数，例如 1、2、4、8。
                - PressureLevels：气压层数。
                - Height：输入特征图的纬向网格数量，例如 91。
                - Width：输入特征图的经向网格数量，例如 180。
                - OutHeight：上采样后的纬向网格数量，例如 91 -> 181。
                - OutWidth：上采样后的经向网格数量，例如 180 -> 360。
                - in_dim：输入token的特征维度，Pangu-Weather中常取 384。
                - out_dim：输出token的特征维度，Pangu-Weather中常取 192。

        Example:
            >>> # Pangu-Weather 中的 surface 分支
            >>> Batch = 2
            >>> input_resolution = (91, 180)
            >>> output_resolution = (181, 360)
            >>> in_dim = 384
            >>> out_dim = 192
            >>> surface_upsample = OneSample(
            ...     style="PanguUpSample",
            ...     input_resolution=input_resolution,
            ...     output_resolution=output_resolution,
            ...     in_dim=in_dim,
            ...     out_dim=out_dim
            ... ).cuda()
            >>> surface_x = torch.randn(Batch, input_resolution[0] * input_resolution[1], in_dim).cuda()
            >>> surface_out = surface_upsample(surface_x)
            >>> surface_out.shape
            torch.Size([2, 65160, 192])

            >>> # Pangu-Weather 中的 upper-air 分支
            >>> Batch = 2
            >>> input_resolution = (8, 91, 180)
            >>> output_resolution = (8, 181, 360)
            >>> in_dim = 384
            >>> out_dim = 192
            >>> upper_air_upsample = OneSample(
            ...     style="PanguUpSample",
            ...     input_resolution=input_resolution,
            ...     output_resolution=output_resolution,
            ...     in_dim=in_dim,
            ...     out_dim=out_dim
            ... ).cuda()
            >>> upper_air_x = torch.randn(
            ...     Batch,
            ...     input_resolution[0] * input_resolution[1] * input_resolution[2],
            ...     in_dim
            ... ).cuda()
            >>> upper_air_out = upper_air_upsample(upper_air_x)
            >>> upper_air_out.shape
            torch.Size([2, 521280, 192])
    """

    def __init__(self, input_resolution, output_resolution, in_dim=384, out_dim=192):
        super().__init__()

        if len(input_resolution) == 2:
            input_resolution = (1, *input_resolution)
        elif len(input_resolution) != 3:
            raise ValueError("input_resolution must have 2 or 3 dimensions")

        if len(output_resolution) == 2:
            output_resolution = (1, *output_resolution)
        elif len(output_resolution) != 3:
            raise ValueError("output_resolution must have 2 or 3 dimensions")

        self.input_resolution = input_resolution
        self.output_resolution = output_resolution
        self.in_dim = in_dim
        self.out_dim = out_dim

        InPressureLevels, InHeight, InWidth = self.input_resolution
        OutPressureLevels, OutHeight, OutWidth = self.output_resolution

        if OutPressureLevels > InPressureLevels:
            raise ValueError("output pressure levels must be less than or equal to input pressure levels")
        if OutHeight > InHeight * 2 or OutWidth > InWidth * 2:
            raise ValueError("output spatial resolution cannot exceed twice the input resolution")

        self.linear1 = nn.Linear(in_dim, out_dim * 4, bias=False)
        self.linear2 = nn.Linear(out_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Batch, NumTokens, InDim = x.shape
        InPressureLevels, InHeight, InWidth = self.input_resolution
        OutPressureLevels, OutHeight, OutWidth = self.output_resolution

        ExpectedTokens = InPressureLevels * InHeight * InWidth
        if NumTokens != ExpectedTokens:
            raise ValueError(f"Expected {ExpectedTokens} tokens, but received {NumTokens}")
        if InDim != self.in_dim:
            raise ValueError(f"Expected input dim {self.in_dim}, but received {InDim}")

        x = self.linear1(x)
        x = x.reshape(Batch, InPressureLevels, InHeight, InWidth, 2, 2, self.out_dim)
        x = x.permute(0, 1, 2, 4, 3, 5, 6)
        x = x.reshape(Batch, InPressureLevels, InHeight * 2, InWidth * 2, self.out_dim)

        HeightPad = InHeight * 2 - OutHeight
        WidthPad = InWidth * 2 - OutWidth

        PaddingTop = HeightPad // 2
        PaddingBottom = HeightPad - PaddingTop
        PaddingLeft = WidthPad // 2
        PaddingRight = WidthPad - PaddingLeft

        x = x[
            :,
            :OutPressureLevels,
            PaddingTop : 2 * InHeight - PaddingBottom,
            PaddingLeft : 2 * InWidth - PaddingRight,
            :,
        ]
        x = x.reshape(Batch, OutPressureLevels * OutHeight * OutWidth, self.out_dim)
        x = self.norm(x)
        x = self.linear2(x)
        return x
