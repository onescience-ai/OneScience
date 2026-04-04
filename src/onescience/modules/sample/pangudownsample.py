import torch
from torch import nn


class PanguDownSample(nn.Module):
    """
        Pangu-Weather模型的统一Down Sample模块。

        Pangu-Weather在多尺度特征编码阶段，使用该模块对token表示进行下采样，
        将相邻网格的局部信息聚合到更低分辨率的特征图中，为后续Transformer/Fuser模块
        提供更大感受野的特征表示。

        输入支持二维和三维输入，会被拆成两条分支：
        - 二维输入（例如地表变量分支）对应的token形状为：
          [Batch, Height * Width, in_dim]
        - 三维输入（例如大气变量分支）对应的token形状为：
          [Batch, PressureLevels * Height * Width, in_dim]

        实现逻辑统一使用三维down sample逻辑：
        - 先根据output_resolution与input_resolution的关系，计算需要补齐的空间padding；
        - 再将输入token恢复为网格形式；
        - 对二维输入，会先在PressureLevels位置补一个长度为1的伪三维维度；
        - 之后仅在Height和Width方向执行2×2邻域聚合；
        - 最后将4个相邻token在特征维拼接，并通过LayerNorm与Linear完成通道投影。

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
                其中Pangu-Weather默认只对Height和Width做2倍下采样，
                PressureLevels维度保持不变。
            in_dim (int):
                输入token的特征维度。
                该模块输出的特征维度为 2 * in_dim。

        形状:
            输入:
                - 二维输入:
                  [Batch, Height * Width, in_dim]
                - 三维输入:
                  [Batch, PressureLevels * Height * Width, in_dim]
            输出:
                - 二维输入对应输出:
                  [Batch, OutHeight * OutWidth, 2 * in_dim]
                - 三维输入对应输出:
                  [Batch, OutPressureLevels * OutHeight * OutWidth, 2 * in_dim]

            各维含义与常见取值：
                - Batch：批大小，即一次前向传播中的样本数，例如 1、2、4、8。
                - PressureLevels：气压层数。
                - Height：输入特征图的纬向网格数量，例如 181。
                - Width：输入特征图的经向网格数量，例如 360。
                - OutHeight：下采样后的纬向网格数量，例如 181 -> 91。
                - OutWidth：下采样后的经向网格数量，例如 360 -> 180。
                - in_dim：输入token的特征维度，Pangu-Weather中常取 192。
                - 2 * in_dim：输出token的特征维度，Pangu-Weather中常取 384。

        Example:
            >>> # Pangu-Weather 中的 surface 分支
            >>> Batch = 2
            >>> input_resolution = (181, 360)
            >>> output_resolution = (91, 180)
            >>> in_dim = 192
            >>> surface_downsample = OneSample(
            ...     style="PanguDownSample",
            ...     input_resolution=input_resolution,
            ...     output_resolution=output_resolution,
            ...     in_dim=in_dim
            ... ).cuda()
            >>> surface_x = torch.randn(Batch, input_resolution[0] * input_resolution[1], in_dim).cuda()
            >>> surface_out = surface_downsample(surface_x)
            >>> surface_out.shape
            torch.Size([2, 16380, 384])

            >>> # Pangu-Weather 中的 upper-air 分支
            >>> Batch = 2
            >>> input_resolution = (8, 181, 360)
            >>> output_resolution = (8, 91, 180)
            >>> in_dim = 192
            >>> upper_air_downsample = OneSample(
            ...     style="PanguDownSample",
            ...     input_resolution=input_resolution,
            ...     output_resolution=output_resolution,
            ...     in_dim=in_dim
            ... ).cuda()
            >>> upper_air_x = torch.randn(
            ...     Batch,
            ...     input_resolution[0] * input_resolution[1] * input_resolution[2],
            ...     in_dim
            ... ).cuda()
            >>> upper_air_out = upper_air_downsample(upper_air_x)
            >>> upper_air_out.shape
            torch.Size([2, 131040, 384])
    """

    def __init__(
        self,
        input_resolution,
        output_resolution,
        in_dim=192,
    ):
        super().__init__()

        if len(input_resolution) == 2:
            input_resolution = (1, *input_resolution)
        if len(output_resolution) == 2:
            output_resolution = (1, *output_resolution)

        self.linear = nn.Linear(in_dim * 4, in_dim * 2, bias=False)
        self.norm = nn.LayerNorm(4 * in_dim)
        self.in_dim = in_dim
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution

        InPressureLevels, InHeight, InWidth = self.input_resolution
        OutPressureLevels, OutHeight, OutWidth = self.output_resolution

        HeightPad = OutHeight * 2 - InHeight
        WidthPad = OutWidth * 2 - InWidth

        PaddingTop = HeightPad // 2
        PaddingBottom = HeightPad - PaddingTop
        PaddingLeft = WidthPad // 2
        PaddingRight = WidthPad - PaddingLeft
        PaddingFront = PaddingBack = 0

        self.pad = nn.ZeroPad3d(
            (PaddingLeft, PaddingRight, PaddingTop, PaddingBottom, PaddingFront, PaddingBack)
        )

    def forward(self, x: torch.Tensor):
        Batch, NumTokens, InDim = x.shape
        InPressureLevels, InHeight, InWidth = self.input_resolution
        OutPressureLevels, OutHeight, OutWidth = self.output_resolution
        ExpectedTokens = InPressureLevels * InHeight * InWidth

        if NumTokens != ExpectedTokens:
            raise ValueError(f"Expected {ExpectedTokens} tokens, but received {NumTokens}")
        if InDim != self.in_dim:
            raise ValueError(f"Expected input dim {self.in_dim}, but received {InDim}")

        x = x.reshape(Batch, InPressureLevels, InHeight, InWidth, InDim)
        x = self.pad(x.permute(0, -1, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        x = x.reshape(
            Batch,
            InPressureLevels,
            OutHeight,
            2,
            OutWidth,
            2,
            InDim,
        ).permute(0, 1, 2, 4, 3, 5, 6)
        x = x.reshape(Batch, OutPressureLevels * OutHeight * OutWidth, 4 * InDim)

        x = self.norm(x)
        x = self.linear(x)
        return x
