from torch import nn
from onescience.modules._lazy import instantiate_registered_style

# 构建统一的池化注册表
_POOLING_REGISTRY = {
    "RNNClusterPooling": (
        "onescience.modules.pooling.rnn_cluster_pooling",
        "RNNClusterPooling",
    ),
}

class OnePooling(nn.Module):

    def __init__(self, style: str, **kwargs):
        super().__init__()

        self.pooler = instantiate_registered_style(
            style,
            _POOLING_REGISTRY,
            "pooling",
            **kwargs,
        )

    def forward(self, *args, **kwargs):
        """
        前向传播
        """
        return self.pooler(*args, **kwargs)
