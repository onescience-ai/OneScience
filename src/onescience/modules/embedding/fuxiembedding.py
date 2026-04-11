from torch import nn


class FuxiEmbedding(nn.Module):
    """
        Fuxi 模型的三维 patch embedding 模块。

        Fuxi 在输入编码阶段，将多时间步二维气象场沿 `TimeSteps` 维堆叠后，
        统一通过 `Conv3d` 切分为三维 patch，并投影到 embedding 特征空间。

        与 Pangu 的 surface / upper-air 双分支不同，Fuxi 的输入是单一路径
        的 `(TimeSteps, Height, Width)` 三维时空块。

        Args:
            img_size (tuple[int, int, int]):
                输入场空间尺寸 `(TimeSteps, Height, Width)`。
            patch_size (tuple[int, int, int]):
                patch 切分尺寸 `(PatchTimeSteps, PatchHeight, PatchWidth)`。
            in_chans (int):
                输入变量通道数。
            embed_dim (int):
                patch embedding 后的输出特征维度。
            norm_layer (nn.Module | None):
                embedding 后的归一化层类型；若为 `None`，则跳过归一化。
            **kwargs:
                额外兼容参数，当前忽略。

        形状:
            输入:
                `x` 形状为 `(Batch, in_chans, TimeSteps, Height, Width)`
            输出:
                `x` 形状为 `(Batch, embed_dim, OutTimeSteps, OutHeight, OutWidth)`

            其中：
            - `OutTimeSteps = TimeSteps // PatchTimeSteps`
            - `OutHeight = Height // PatchHeight`
            - `OutWidth = Width // PatchWidth`

            补充说明：
            - 当前实现不做自动 padding
            - 若尺寸不能被整除，尾部剩余区域不会进入输出特征图

        Example:
            >>> Batch = 2
            >>> Variables = 70
            >>> TimeSteps = 2
            >>> Height = 721
            >>> Width = 1440
            >>> embedding = FuxiEmbedding(
            ...     img_size=(TimeSteps, Height, Width),
            ...     patch_size=(2, 4, 4),
            ...     in_chans=Variables,
            ...     embed_dim=1536,
            ... )
            >>> x = torch.randn(Batch, Variables, TimeSteps, Height, Width)
            >>> out = embedding(x)
            >>> out.shape
            torch.Size([2, 1536, 1, 180, 360])
    """

    def __init__(
        self,
        img_size=(2, 721, 1440),
        patch_size=(2, 4, 4),
        in_chans=70,
        embed_dim=1536,
        norm_layer=nn.LayerNorm,
        **kwargs,
    ):
        super().__init__()

        TimeSteps, Height, Width = img_size
        PatchTimeSteps, PatchHeight, PatchWidth = patch_size
        patches_resolution = [
            TimeSteps // PatchTimeSteps,
            Height // PatchHeight,
            Width // PatchWidth,
        ]

        self.img_size = img_size
        self.patches_resolution = patches_resolution
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        Batch, _, TimeSteps, Height, Width = x.shape
        ExpectedTimeSteps, ExpectedHeight, ExpectedWidth = self.img_size

        if (
            TimeSteps != ExpectedTimeSteps
            or Height != ExpectedHeight
            or Width != ExpectedWidth
        ):
            raise ValueError(
                f"Input image size ({TimeSteps}*{Height}*{Width}) does not match "
                f"configured size ({ExpectedTimeSteps}*{ExpectedHeight}*{ExpectedWidth})"
            )

        x = self.proj(x).reshape(Batch, self.embed_dim, -1).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        x = x.transpose(1, 2).reshape(Batch, self.embed_dim, *self.patches_resolution)
        return x
