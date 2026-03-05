from typing import Union

import torch
import torch.nn as nn
from dgl import DGLGraph
from torch import Tensor

from onescience.modules.mlp.mesh_graph_mlp import MeshGraphEdgeMLPConcat, MeshGraphEdgeMLPSum, MeshGraphMLP
from onescience.modules.utils.gnnlayer_utils import CuGraphCSC, aggregate_and_concat

class MeshGraphDecoder(nn.Module):
    """
    用于 GraphCast 或 MeshGraphNet 等模型中的解码器模块。

    它作用于连接网格（Mesh，通常代表隐空间）和栅格（Grid，通常代表输出域）的二部图（Bipartite Graph）。
    该模块负责将信息从网格传递回栅格，从而从隐空间恢复出物理空间的预测结果。

    其主要计算流程包括：
        1. 更新边特征 (Edge Update): 根据输入的边特征、源节点（Mesh）特征和目标节点（Grid）特征，利用 MLP 更新边特征。
        2. 消息聚合 (Message Aggregation): 将更新后的边特征聚合（求和或求平均）到对应的目标节点（Grid）上。
        3. 更新节点特征 (Node Update): 将聚合后的信息与目标节点的原始特征拼接，通过节点 MLP 进行变换，并应用残差连接得到最终的 Grid 节点特征。

    Args:
        aggregation (str, optional): 消息聚合方法，可选 "sum" 或 "mean"。默认值: "sum"。
        input_dim_src_nodes (int, optional): 输入源节点（Mesh）特征的维度。默认值: 512。
        input_dim_dst_nodes (int, optional): 输入目标节点（Grid）特征的维度。默认值: 512。
        input_dim_edges (int, optional): 输入边特征的维度。默认值: 512。
        output_dim_dst_nodes (int, optional): 输出目标节点（Grid）特征的维度。默认值: 512。
        output_dim_edges (int, optional): 输出边特征的维度。默认值: 512。
        hidden_dim (int, optional): MLP 隐藏层的神经元数量。默认值: 512。
        hidden_layers (int, optional): 隐藏层的层数。默认值: 1。
        activation_fn (nn.Module, optional): 激活函数类型。默认值: nn.SiLU()。
        norm_type (str, optional): 归一化类型 ("LayerNorm" 或 "TELayerNorm")。默认值: "LayerNorm"。
        do_concat_trick (bool, optional): 是否使用“拼接技巧”优化显存（参见 MeshEdgeBlock）。默认值: False。
        recompute_activation (bool, optional): 是否启用激活重计算以节省显存。默认值: False。

    形状:
        输入 m2g_efeat: (E, C_edge_in)，Mesh-to-Grid 边的特征。
        输入 grid_nfeat: (N_grid, C_grid_in)，Grid 节点的特征（目标节点）。
        输入 mesh_nfeat: (N_mesh, C_mesh_in)，Mesh 节点的特征（源节点）。
        输入 graph: DGLGraph 或 CuGraphCSC 对象，表示 Mesh 到 Grid 的二部图。
        输出: (N_grid, C_grid_out)，更新后的 Grid 节点特征。

    Example:
        >>> # 假设 Mesh 有 200 个节点，Grid 有 1000 个节点，之间有 3000 条边
        >>> decoder = MeshGraphDecoder(
        ...     aggregation="mean",
        ...     input_dim_src_nodes=64,
        ...     input_dim_dst_nodes=64,
        ...     input_dim_edges=32,
        ...     output_dim_dst_nodes=64,
        ...     output_dim_edges=32,
        ...     hidden_dim=128
        ... )
        >>> m2g_efeat = torch.randn(3000, 32)
        >>> grid_nfeat = torch.randn(1000, 64)
        >>> mesh_nfeat = torch.randn(200, 64)
        >>> # graph 为预定义的 Mesh-to-Grid 二部图对象
        >>> grid_out = decoder(m2g_efeat, grid_nfeat, mesh_nfeat, graph)
        >>> grid_out.shape
        torch.Size([1000, 64])
    """

    def __init__(
        self,
        aggregation: str = "sum",
        input_dim_src_nodes: int = 512,
        input_dim_dst_nodes: int = 512,
        input_dim_edges: int = 512,
        output_dim_dst_nodes: int = 512,
        output_dim_edges: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        do_concat_trick: bool = False,
        recompute_activation: bool = False,
    ):
        super().__init__()
        self.aggregation = aggregation

        MLP = MeshGraphEdgeMLPSum if do_concat_trick else MeshGraphEdgeMLPConcat
        # edge MLP
        self.edge_mlp = MLP(
            efeat_dim=input_dim_edges,
            src_dim=input_dim_src_nodes,
            dst_dim=input_dim_dst_nodes,
            output_dim=output_dim_edges,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

        # dst node MLP
        self.node_mlp = MeshGraphMLP(
            input_dim=input_dim_dst_nodes + output_dim_edges,
            output_dim=output_dim_dst_nodes,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

    @torch.jit.ignore()
    def forward(
        self,
        m2g_efeat: Tensor,
        grid_nfeat: Tensor,
        mesh_nfeat: Tensor,
        graph: Union[DGLGraph, CuGraphCSC],
    ) -> Tensor:
        # update edge features
        efeat = self.edge_mlp(m2g_efeat, (mesh_nfeat, grid_nfeat), graph)
        # aggregate messages (edge features) to obtain updated node features
        cat_feat = aggregate_and_concat(efeat, grid_nfeat, graph, self.aggregation)
        # transformation and residual connection
        dst_feat = self.node_mlp(cat_feat) + grid_nfeat
        return dst_feat
