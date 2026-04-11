import torch.nn as nn


class FourCastNetEmbedding(nn.Module):
    """
        FourCastNet 的 2D Patch Embedding 模块。

        FourCastNet 在输入编码阶段，使用该模块将二维气象场切分为不重叠 patch，
        并将每个 patch 投影到统一的 embedding 特征空间。

        该模块只处理二维输入，不涉及三维 `PressureLevels` 维度。
        输出为展平后的 patch token 序列，后续会在调用层中恢复成
        `(PatchGridHeight, PatchGridWidth)` 网格，再送入 AFNO 主干。

        Args:
            img_size (tuple[int, int]):
                输入场空间尺寸 `(Height, Width)`。
            patch_size (tuple[int, int]):
                patch 切分尺寸 `(PatchHeight, PatchWidth)`。
            in_chans (int):
                输入变量通道数。
            embed_dim (int):
                patch embedding 后的输出特征维度。

        形状:
            输入:
                `x` 形状为 `(Batch, Channels, Height, Width)`
            输出:
                `x` 形状为 `(Batch, NumPatches, embed_dim)`

            其中：
            - `PatchGridHeight = Height // PatchHeight`
            - `PatchGridWidth = Width // PatchWidth`
            - `NumPatches = PatchGridHeight * PatchGridWidth`

        Example:
            >>> Batch = 2
            >>> Channels = 20
            >>> Height = 720
            >>> Width = 1440
            >>> embedding = FourCastNetEmbedding(
            ...     img_size=(Height, Width),
            ...     patch_size=(8, 8),
            ...     in_chans=Channels,
            ...     embed_dim=768,
            ... )
            >>> x = torch.randn(Batch, Channels, Height, Width)
            >>> out = embedding(x)
            >>> out.shape
            torch.Size([2, 16200, 768])
    """

    def __init__(
        self,
        img_size=(720, 1440),
        patch_size=(8, 8),
        in_chans=19,
        embed_dim=768,
    ):
        super().__init__()

        Height, Width = img_size
        PatchHeight, PatchWidth = patch_size

        PatchGridHeight = Height // PatchHeight
        PatchGridWidth = Width // PatchWidth
        num_patches = PatchGridHeight * PatchGridWidth

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        _, _, Height, Width = x.shape
        ExpectedHeight, ExpectedWidth = self.img_size

        if Height != ExpectedHeight or Width != ExpectedWidth:
            raise ValueError(
                f"Input image size ({Height}*{Width}) does not match "
                f"configured size ({ExpectedHeight}*{ExpectedWidth})"
            )

        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
