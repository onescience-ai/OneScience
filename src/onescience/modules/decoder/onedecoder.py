import torch
from torch import nn

# 从统一的解码器文件导入具体实现
from .unet_decoder import UNetDecoder1D, UNetDecoder2D, UNetDecoder3D
from .graphvit_decoder import GraphViTDecoder
from .mesh_graph_decoder import MeshGraphDecoder
# 构建统一的注册表
_DECODER_REGISTRY = {
    "UNetDecoder1D": UNetDecoder1D,
    "UNetDecoder2D": UNetDecoder2D,
    "UNetDecoder3D": UNetDecoder3D,
    "GraphViTDecoder": GraphViTDecoder,
    "MeshGraphDecoder": MeshGraphDecoder,
}

class OneDecoder(nn.Module):
    """
    OneDecoder 统一解码器调用接口。
    """
    def __init__(self, style: str, **kwargs):
        super().__init__()

        if style not in _DECODER_REGISTRY:
            raise NotImplementedError(
                f"Unknown style: '{style}'. Available options are: {list(_DECODER_REGISTRY.keys())}"
            )
        
        # 实例化具体的解码器层
        self.decoder = _DECODER_REGISTRY[style](**kwargs)

    def forward(self, *args, **kwargs):
        """
        前向传播。
        使用 *args 和 **kwargs 可以无缝透传给底层的具体实现模块。
        """
        return self.decoder(*args, **kwargs)