import torch
from torch import nn
from torch.nn import functional as F

from onescience.modules import OneEmbedding, OneTransformer, OneFC


class Fuxi(nn.Module):
    """
    Fuxi 的主模型实现。

    该模型使用以下组件完成输入编码、二维 trunk 特征提取与 patch 级输出恢复：

    - `OneEmbedding(style="FuxiEmbedding")`
      - 将 `(TimeSteps, Height, Width)` 三维时空块映射为 patch 特征
    - `OneTransformer(style="FuxiTransformer")`
      - 在二维特征图上执行下采样、Swin trunk、上采样
    - `OneFC(style="FuxiFC")`
      - 将每个二维网格位置的 embedding 特征映射为 patch 级输出变量

    在当前实现中：

    - 输入包含多个时间步的二维气象场
    - `patch_size[0]` 默认与 `TimeSteps` 相同，使 embedding 后时间维压缩为 1
    - trunk 只处理二维特征图
    - 最终通过 patch 重排与双线性插值恢复到目标空间分辨率

    Args:
        img_size (tuple[int, int, int]):
            输入空间尺寸 `(TimeSteps, Height, Width)`。
        patch_size (tuple[int, int, int]):
            patch 切分尺寸 `(PatchTimeSteps, PatchHeight, PatchWidth)`。
        in_chans (int):
            输入变量通道数。
        out_chans (int):
            输出变量通道数。
        embed_dim (int):
            embedding 特征维度。
        num_groups (int):
            trunk 中采样模块的 `GroupNorm` 分组数。
        num_heads (int):
            `SwinTransformerV2Stage` 的注意力头数。
        window_size (int | tuple[int, int]):
            trunk 局部窗口大小。
    """

    def __init__(
        self,
        img_size=(2, 721, 1440),
        patch_size=(2, 4, 4),
        in_chans=70,
        out_chans=70,
        embed_dim=1536,
        num_groups=32,
        num_heads=8,
        window_size=7,
    ):
        super().__init__()

        TimeSteps, Height, Width = img_size
        PatchTimeSteps, PatchHeight, PatchWidth = patch_size
        if TimeSteps != PatchTimeSteps:
            raise ValueError(
                "Current Fuxi model expects patch_size[0] to equal img_size[0] "
                "so the embedding output time dimension is 1 before squeeze"
            )

        EmbeddedHeight = Height // PatchHeight
        EmbeddedWidth = Width // PatchWidth
        TransformerInputResolution = (
            EmbeddedHeight // 2,
            EmbeddedWidth // 2,
        )

        self.cube_embedding = OneEmbedding(
            style="FuxiEmbedding",
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.u_transformer = OneTransformer(
            style="FuxiTransformer",
            embed_dim=embed_dim,
            num_groups=num_groups,
            input_resolution=TransformerInputResolution,
            num_heads=num_heads,
            window_size=window_size,
        )
        self.fc = OneFC(
            style="FuxiFC",
            in_channels=embed_dim,
            out_channels=out_chans * PatchHeight * PatchWidth,
        )

        self.patch_size = patch_size
        self.transformer_input_resolution = TransformerInputResolution
        self.embedded_resolution = (EmbeddedHeight, EmbeddedWidth)
        self.out_chans = out_chans
        self.img_size = img_size

    def forward(self, x):
        """
        Args:
            x (torch.Tensor):
                输入张量，形状为 `(Batch, in_chans, TimeSteps, Height, Width)`。

        Returns:
            torch.Tensor:
                输出张量，形状为 `(Batch, out_chans, Height, Width)`。
        """
        Batch, _, _, _, _ = x.shape
        _, PatchHeight, PatchWidth = self.patch_size
        EmbeddedHeight, EmbeddedWidth = self.embedded_resolution

        x = self.cube_embedding(x)
        if x.shape[2] != 1:
            raise ValueError(
                f"Expected embedding time dimension 1 before squeeze, but received {x.shape[2]}"
            )

        x = x.squeeze(2)
        x = self.u_transformer(x)
        x = self.fc(x.permute(0, 2, 3, 1))
        x = x.reshape(
            Batch,
            EmbeddedHeight,
            EmbeddedWidth,
            PatchHeight,
            PatchWidth,
            self.out_chans,
        ).permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(
            Batch,
            EmbeddedHeight * PatchHeight,
            EmbeddedWidth * PatchWidth,
            self.out_chans,
        )
        x = x.permute(0, 3, 1, 2)

        x = F.interpolate(x, size=self.img_size[1:], mode="bilinear")

        return x
