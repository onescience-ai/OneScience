import torch
from torch import nn


class PanguEmbedding(nn.Module):
    """
        Pangu-Weather模型的统一Patch Embedding模块。

        Pangu-Weather的输入特征编码阶段，负责将原始气象场切分为非重叠patch，
        并将每个patch投影到统一的嵌入特征空间，为后续Transformer/Fuser模块提供基础特征表示。

        输入支持二维和三维输入，会被拆成两条分支：
        - 二维输入（例如地表变量、静态变量等）形状为：
          (Batch, Variables, Height, Width)
        - 三维输入（例如大气变量）形状为：
          (Batch, Variables, Pressure Levels, Height, Width)

        实现逻辑统一使用三维patch embedding逻辑：
        - 先根据patch_size对输入做零填充，使各维长度可被整除；
        - 再使用Conv3d，patch划分与线性投影；
        - 若输入是二维张量，则先在Pressure Levels位置补一个长度为1的伪三维维度，完成三维投影后再将该维度去掉。

        因此，该模块同时支持二维输入和三维输入，但内部始终走统一的三维实现。

        Args:
            img_size (tuple[int, int] | tuple[int, int, int]):
                输入场空间尺寸。
                - 二维输入对应 (Height, Width)
                - 三维输入对应 (Pressure Levels, Height, Width)
            patch_size (tuple[int, int] | tuple[int, int, int]):
                patch 的切分尺寸。
                - 二维输入对应 (Patch Height, Patch Width)
                - 三维输入对应 (Patch Pressure Levels, Patch Height, Patch Width)
            Variables (int):
                输入变量通道数。
                - 默认Pangu模型二维输入通常为7=4个地陆地变量+3个静态掩码
                - 默认Pangu模型三维输入通常为5个大气变量对应 Z、Q、T、U、V
            embed_dim (int):
                patch embedding投影后的输出特征通道数，类似大语言模型的词嵌入。
            norm_layer (nn.Module, optional):
                投影后使用的归一化层，默认为 None。常用: nn.LayerNorm。

        形状:
            输入:
                - 二维输入:
                  [Batch, Variables, Height, Width]
                - 三维输入:
                  [Batch, Variables, Pressure Levels, Height, Width]
            输出:
                - 二维输入对应输出:
                  [Batch, embed_dim, Out Height, Out Width]
                - 三维输入对应输出:
                  [Batch, embed_dim, Out Pressure Levels, Out Height, Out Width]

                其中：
                - Out Pressure Levels = ceil(Pressure Levels / Patch Pressure Levels)
                - Out Height = ceil(Height / Patch Height)
                - Out Width = ceil(Width / Patch Width)

            各维含义与常见取值：
                - Batch：批大小，即一次前向传播中的样本数，例如 1、2、4、8。
                - Variables：输入变量数。
                - Pressure Levels：气压层数。
                - Height：气象变量表达为图像时纬度网格数量，常取为721。
                - Width：气象变量表达为图像时经度网格数量，常取为1440。
                - embed_dim：patch embedding后每个网格的特征数量，默认为192。

        Example:
            >>> # Pangu-Weather 中的 surface 分支
            >>> # 输入来自 4 个地表变量 + 3 个静态掩码
            >>> Batch = 2
            >>> Variables = 7
            >>> img_size = [721, 1440]
            >>> patch_size = (4, 4)
            >>> embed_dim = 192
            >>> surface_patch_embed = OneEmbedding(
            ...     style="PanguEmbedding",
            ...     img_size=img_size,
            ...     patch_size=patch_size,
            ...     Variables=Variables,
            ...     embed_dim=embed_dim
            ... ).cuda()
            >>> surface_x = torch.randn(Batch, Variables, *img_size).cuda()
            >>> surface_out = surface_patch_embed(surface_x)
            >>> surface_out.shape
            torch.Size([2, 192, 181, 360])

            >>> # Pangu-Weather 中的 upper-air 分支
            >>> # 原始 65 个高空通道会先 reshape 成 5 个变量类型 × 13 个气压层
            >>> Batch = 2
            >>> Variables = 5
            >>> img_size = [13, 721, 1440]
            >>> patch_size = (2, 4, 4)
            >>> embed_dim = 192
            >>> upper_air_patch_embed = OneEmbedding(
            ...     style="PanguEmbedding",
            ...     img_size=img_size,
            ...     patch_size=patch_size,
            ...     Variables=Variables,
            ...     embed_dim=embed_dim
            ... ).cuda()
            >>> upper_air_x = torch.randn(Batch, Variables, *img_size).cuda()
            >>> upper_air_out = upper_air_patch_embed(upper_air_x)
            >>> upper_air_out.shape
            torch.Size([2, 192, 7, 181, 360])
    """

    def __init__(
        self,
        img_size=(13, 721, 1440),
        patch_size=(2, 4, 4),
        Variables=5,
        embed_dim=192,
        norm_layer=None,
    ):
        super().__init__()

        if len(img_size) == 2:
            img_size = (1, *img_size)
        if len(patch_size) == 2:
            patch_size = (1, *patch_size)

        level, height, width = img_size
        l_patch_size, h_patch_size, w_patch_size = patch_size

        padding_left = padding_right = padding_top = padding_bottom = padding_front = padding_back = 0

        l_remainder = level % l_patch_size
        h_remainder = height % h_patch_size
        w_remainder = width % w_patch_size

        if l_remainder:
            l_pad = l_patch_size - l_remainder
            padding_front = l_pad // 2
            padding_back = l_pad - padding_front
        if h_remainder:
            h_pad = h_patch_size - h_remainder
            padding_top = h_pad // 2
            padding_bottom = h_pad - padding_top
        if w_remainder:
            w_pad = w_patch_size - w_remainder
            padding_left = w_pad // 2
            padding_right = w_pad - padding_left

        self.pad = nn.ZeroPad3d(
            (
                padding_left,
                padding_right,
                padding_top,
                padding_bottom,
                padding_front,
                padding_back,
            )
        )
        self.proj = nn.Conv3d(
            Variables, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor):
        squeeze_level_dim = False
        if x.ndim == 4:
            x = x.unsqueeze(2)
            squeeze_level_dim = True

        x = self.pad(x)
        x = self.proj(x)
        if self.norm:
            x = self.norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        if squeeze_level_dim:
            x = x.squeeze(2)
        return x
