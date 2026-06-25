import torch.nn as nn

from onescience.modules._lazy import instantiate_registered_style

_EDGE_REGISTRY = {
    "MeshEdgeBlock": ("onescience.modules.edge.mesh_edge_block", "MeshEdgeBlock"),
    "MeshEdgeDistributedBlock": ("onescience.modules.edge.mesh_edge_distributed_block", "MeshEdgeDistributedBlock"),
}

class OneEdge(nn.Module):
    """
    OneEdge: 统一边更新模块调用接口。
    
    负责实例化各种边更新策略。通常用于 GNN 的 Message Passing 阶段。
    """
    def __init__(self, style: str, **kwargs):
        super().__init__()
        self.edge_updater = instantiate_registered_style(style, _EDGE_REGISTRY, "edge", **kwargs)

    def forward(self, *args, **kwargs):
        """
        透传参数 (通常是 efeat, nfeat, graph)
        """
        return self.edge_updater(*args, **kwargs)
