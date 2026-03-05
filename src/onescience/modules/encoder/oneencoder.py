import torch
from torch import nn

from .unet_encoder import UNetEncoder1D, UNetEncoder2D, UNetEncoder3D
from .graphvit_encoder import GraphViTEncoder
from .mesh_graph_encoder import MeshGraphEncoder
# 构建统一的注册表
_ENCODER_REGISTRY = {
    "UNetEncoder1D": UNetEncoder1D,
    "UNetEncoder2D": UNetEncoder2D,
    "UNetEncoder3D": UNetEncoder3D,
    "GraphViTEncoder": GraphViTEncoder,
    "MeshGraphEncoder":MeshGraphEncoder,
}

class OneEncoder(nn.Module):
    """
    OneEncoder 统一编码器调用接口。
    """
    def __init__(self, style: str, **kwargs):
        super().__init__()

        if style not in _ENCODER_REGISTRY:
            raise NotImplementedError(
                f"Unknown style: '{style}'. Available options are: {list(_ENCODER_REGISTRY.keys())}"
            )
        
        # 实例化具体的编码器层
        self.encoder = _ENCODER_REGISTRY[style](**kwargs)

    def forward(self, *args, **kwargs):
        """
        前向传播。
        使用 *args 和 **kwargs 可以无缝透传给底层的具体实现模块。
        """
        return self.encoder(*args, **kwargs)