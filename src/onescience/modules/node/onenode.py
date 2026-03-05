import torch
import torch.nn as nn

# 导入具体的节点更新实现
from .mesh_node_block import MeshNodeBlock

_NODE_REGISTRY = {
    "MeshNodeBlock": MeshNodeBlock,
}

class OneNode(nn.Module):
    """
    OneNode: 统一节点更新模块调用接口。
    
    负责实例化各种节点更新策略。通常用于 GNN 的 Node Update 阶段。
    """
    def __init__(self, style: str, **kwargs):
        super().__init__()
        if style not in _NODE_REGISTRY:
            raise NotImplementedError(
                f"Unknown node style: '{style}'. Available: {list(_NODE_REGISTRY.keys())}"
            )
        
        self.node_updater = _NODE_REGISTRY[style](**kwargs)

    def forward(self, *args, **kwargs):
        """
        透传参数 (通常是 efeat, nfeat, graph)
        """
        return self.node_updater(*args, **kwargs)