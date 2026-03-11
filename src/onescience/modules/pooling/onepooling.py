import torch
from torch import nn
from .rnn_cluster_pooling import RNNClusterPooling

# 构建统一的池化注册表
_POOLING_REGISTRY = {
    "RNNClusterPooling": RNNClusterPooling,

}

class OnePooling(nn.Module):

    def __init__(self, style: str, **kwargs):
        super().__init__()

        if style not in _POOLING_REGISTRY:
            raise NotImplementedError(
                f"Unknown style: '{style}'. Available options are: {list(_POOLING_REGISTRY.keys())}"
            )
        
        # 实例化具体的池化层
        self.pooler = _POOLING_REGISTRY[style](**kwargs)

    def forward(self, *args, **kwargs):
        """
        前向传播
        """
        return self.pooler(*args, **kwargs)