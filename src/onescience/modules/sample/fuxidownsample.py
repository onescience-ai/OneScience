from torch import nn


class FuxiDownSample(nn.Module):
    """
        Fuxi 模型的二维下采样模块。

        该模块先用步长为 2 的卷积将空间分辨率减半，再通过若干残差卷积块
        对下采样后的特征做局部细化。

        它处理的是二维特征图，不是 token 序列。

        Args:
            in_chans (int):
                输入特征通道数。
            out_chans (int):
                输出特征通道数。
            num_groups (int):
                `GroupNorm` 的分组数。
            num_residuals (int):
                残差卷积块数量。

        形状:
            输入:
                `x` 形状为 `(Batch, in_chans, Height, Width)`
            输出:
                `x` 形状为 `(Batch, out_chans, OutHeight, OutWidth)`

            其中：
            - `OutHeight = Height // 2`
            - `OutWidth = Width // 2`
            - 若输入为奇数尺寸，尾部多出的 1 行或 1 列会在输出中裁掉

        Example:
            >>> downsample = FuxiDownSample(
            ...     in_chans=1536,
            ...     out_chans=1536,
            ...     num_groups=32,
            ...     num_residuals=2,
            ... )
            >>> x = torch.randn(2, 1536, 180, 360)
            >>> out = downsample(x)
            >>> out.shape
            torch.Size([2, 1536, 90, 180])
    """

    def __init__(
        self,
        in_chans=1536,
        out_chans=1536,
        num_groups=32,
        num_residuals=2,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=(3, 3), stride=2, padding=1)

        blocks = []
        for _ in range(num_residuals):
            blocks.append(nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1))
            blocks.append(nn.GroupNorm(num_groups, out_chans))
            blocks.append(nn.SiLU())

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        _, _, Height, Width = x.shape
        x = self.conv(x)

        Shortcut = x

        x = self.blocks(x)

        Output = x + Shortcut
        if Height % 2 != 0:
            Output = Output[:, :, :-1, :]
        if Width % 2 != 0:
            Output = Output[:, :, :, :-1]
        return Output
