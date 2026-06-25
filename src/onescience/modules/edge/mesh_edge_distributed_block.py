from typing import Union

import torch
import torch.nn as nn
from dgl import DGLGraph
from torch import Tensor

from onescience.modules.mlp.mesh_graph_distributed_mlp import DistributedMeshGraphEdgeMLPConcat, DistributedMeshGraphEdgeMLPSum
from onescience.modules.utils.gnnlayer_utils import CuGraphCSC


class DistributedMeshEdgeBlock(nn.Module):
    """
    用于 GraphCast 或 MeshGraphNet 等模型中的边更新块 (Edge Block)。

    该模块在网格 (Mesh) 表示的隐空间上运行，负责根据当前的边特征以及连接该边的源节点和目标节点的特征来更新边特征。
    模块内部包含一个 MLP，并采用了残差连接 (Residual Connection)，即输出特征是 MLP 的变换结果与输入边特征之和。

    Args:
        input_dim_nodes (int, optional): 输入节点特征的维度。默认值: 512。
        input_dim_edges (int, optional): 输入边特征的维度。默认值: 512。
        output_dim (int, optional): 输出边特征的维度。为了支持残差连接，通常应与 input_dim_edges 保持一致。默认值: 512。
        hidden_dim (int, optional): MLP 隐藏层的神经元数量。默认值: 512。
        hidden_layers (int, optional): MLP 中隐藏层的层数。默认值: 1。
        activation_fn (nn.Module, optional): 激活函数类型。默认值: nn.SiLU()。
        norm_type (str, optional): 归一化类型 ("LayerNorm" 或 "TELayerNorm")。默认值: "LayerNorm"。
        do_concat_trick (bool, optional): 是否使用“拼接技巧”优化显存。
            如果为 True，使用 MeshGraphEdgeMLPSum（分别线性变换后相加），可节省显存。
            如果为 False，使用 MeshGraphEdgeMLPConcat（先拼接特征再变换）。
            默认值: False。
        recompute_activation (bool, optional): 是否启用激活重计算以节省显存（仅支持 SiLU）。默认值: False。

    形状:
        输入 efeat: (E, C_edge_in)，其中 E 是边的数量。
        输入 nfeat: (N, C_node)，其中 N 是节点的数量。
        输入 graph: DGLGraph 或 CuGraphCSC 对象，定义图的拓扑结构。
        输出: 返回一个元组 (new_efeat, nfeat)。
            new_efeat: (E, C_edge_out)，更新后的边特征。
            nfeat: (N, C_node)，原样返回的节点特征。

    Example:
        >>> # 假设有 100 个节点，500 条边，特征维度均为 64
        >>> edge_block = MeshEdgeBlock(
        ...     input_dim_nodes=64,
        ...     input_dim_edges=64,
        ...     output_dim=64,
        ...     hidden_dim=128,
        ...     do_concat_trick=True
        ... )
        >>> efeat = torch.randn(500, 64)
        >>> nfeat = torch.randn(100, 64)
        >>> # graph 为图结构对象
        >>> new_efeat, _ = edge_block(efeat, nfeat, graph)
        >>> new_efeat.shape
        torch.Size([500, 64])
    """

    def __init__(
        self,
        input_dim_nodes: int = 512,
        input_dim_edges: int = 512,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        do_concat_trick: bool = False,
        recompute_activation: bool = False,
        config=None,
    ):
        super().__init__()
        self.config = config

        MLP = DistributedMeshGraphEdgeMLPSum if do_concat_trick else DistributedMeshGraphEdgeMLPConcat

        self.edge_mlp = MLP(
            efeat_dim=input_dim_edges,
            src_dim=input_dim_nodes,
            dst_dim=input_dim_nodes,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
            config=config,
        )

    @torch.jit.ignore()
    def forward(
        self,
        efeat: Tensor,
        nfeat: Tensor,
        graph: Union[DGLGraph, CuGraphCSC],
    ) -> Tensor:
        # print(f"[DistributedMeshEdgeBlock forward] 输入 efeat.shape = {efeat.shape}, nfeat.shape = {nfeat.shape}")
        efeat_new = self.edge_mlp(efeat, nfeat, graph)
        efeat_new = efeat_new + efeat
        # print(f"[DistributedMeshEdgeBlock forward] 输出 efeat_new.shape = {efeat_new.shape}")
        return efeat_new, nfeat