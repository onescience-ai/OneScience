from torch import nn


class FourCastNetFC(nn.Module):
    """
        FourCastNet 中使用的逐位置前馈网络模块。

        该模块只在最后一个特征维度上做两层全连接变换，不改变前面的
        batch 或空间网格维度。它通常位于 `FourCastNetFuser` 中的 AFNO
        频域混合之后，用于做通道混合。

        Args:
            in_features (int):
                输入特征维度。
            hidden_features (int | None):
                中间隐藏层维度；若为 `None`，则退回到 `in_features`。
            out_features (int | None):
                输出特征维度；若为 `None`，则退回到 `in_features`。
            act_layer (nn.Module):
                激活函数类型。
            drop (float):
                dropout 比例。

        形状:
            输入:
                `x` 形状为 `(..., in_features)`
            输出:
                `x` 形状为 `(..., out_features)`
    """

    def __init__(
        self,
        in_features=768,
        hidden_features=3072,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
