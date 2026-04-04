from torch import nn


class FuxiFC(nn.Module):
    """
        Fuxi 模型中的逐位置线性投影模块。

        该模块作用在最后一个特征维度上，将 trunk 输出的 embedding 特征
        直接映射为 patch 级输出变量。

        它不改变前面的 batch 或空间网格维度。

        Args:
            in_channels (int):
                输入特征维度。
            out_channels (int):
                输出特征维度。

        形状:
            输入:
                `x` 形状为 `(..., in_channels)`
            输出:
                `x` 形状为 `(..., out_channels)`
    """

    def __init__(
        self,
        in_channels=1536,
        out_channels=70 * 4 * 4,
    ):
        super().__init__()

        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        x = self.fc(x)
        return x
