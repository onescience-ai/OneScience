import torch.nn as nn

from onescience.modules._lazy import instantiate_registered_style

_NODE_REGISTRY = {
    "MeshNodeBlock": ("onescience.modules.node.mesh_node_block", "MeshNodeBlock"),
}

class OneNode(nn.Module):
    """
    OneNode: 统一节点更新模块调用接口。
    
    负责实例化各种节点更新策略。通常用于 GNN 的 Node Update 阶段。
    """
    def __init__(self, style: str, **kwargs):
        super().__init__()
        self.node_updater = instantiate_registered_style(style, _NODE_REGISTRY, "node", **kwargs)

    def forward(self, *args, **kwargs):
        """
        透传参数 (通常是 efeat, nfeat, graph)
        """
        return self.node_updater(*args, **kwargs)
