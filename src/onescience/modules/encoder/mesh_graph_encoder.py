from typing import Tuple, Union

import torch
import torch.nn as nn
from dgl import DGLGraph
from torch import Tensor

from .mesh_graph_mlp import MeshGraphEdgeMLPConcat, MeshGraphEdgeMLPSum, MeshGraphMLP
from .utils import CuGraphCSC, aggregate_and_concat


class MeshGraphEncoder(nn.Module):
    """
    用于 GraphCast 或 MeshGraphNet 等模型中的编码器模块。

    该模块作用于连接规则网格（Grid，代表输入域）和网格（Mesh，代表隐空间）的二部图（Bipartite Graph）。
    它负责将信息从输入网格（源节点）编码并传递到隐空间网格（目标节点）。

    计算流程包括：
        1. 边特征更新: 利用 MLP 结合当前的边特征、Grid 节点特征和 Mesh 节点特征来更新 Grid-to-Mesh 的边特征。
        2. 消息聚合: 将更新后的边特征聚合到 Mesh 节点（目标节点），并与 Mesh 原始特征拼接。
        3. 节点特征更新:
            Mesh 节点: 通过 MLP 处理聚合后的特征，并应用残差连接更新 Mesh 节点特征。
            Grid 节点: 通过一个独立的 MLP 对 Grid 节点特征进行变换，并应用残差连接（虽然主要目的是编码到 Mesh，但 Grid 节点特征也会经过一层处理）。

    Args:
        aggregation (str, optional): 消息聚合方法，可选 "sum" 或 "mean"。默认值: "sum"。
        input_dim_src_nodes (int, optional): 输入源节点（Grid）特征的维度。默认值: 512。
        input_dim_dst_nodes (int, optional): 输入目标节点（Mesh）特征的维度。默认值: 512。
        input_dim_edges (int, optional): 输入边特征的维度。默认值: 512。
        output_dim_src_nodes (int, optional): 输出源节点（Grid）特征的维度。默认值: 512。
        output_dim_dst_nodes (int, optional): 输出目标节点（Mesh）特征的维度。默认值: 512。
        output_dim_edges (int, optional): 输出边特征的维度。默认值: 512。
        hidden_dim (int, optional): MLP 隐藏层的神经元数量。默认值: 512。
        hidden_layers (int, optional): 隐藏层的层数。默认值: 1。
        activation_fn (nn.Module, optional): 激活函数类型。默认值: nn.SiLU()。
        norm_type (str, optional): 归一化类型 ("LayerNorm" 或 "TELayerNorm")。默认值: "LayerNorm"。
        do_concat_trick (bool, optional): 是否使用“拼接技巧”优化显存。默认值: False。
        recompute_activation (bool, optional): 是否启用激活重计算以节省显存。默认值: False。

    形状:
        输入 g2m_efeat: (E, C_edge_in)，Grid-to-Mesh 边的特征。
        输入 grid_nfeat: (N_grid, C_src_in)，Grid 节点的特征（源节点）。
        输入 mesh_nfeat: (N_mesh, C_dst_in)，Mesh 节点的特征（目标节点）。
        输入 graph: DGLGraph 或 CuGraphCSC 对象，表示 Grid 到 Mesh 的二部图。
        输出: 返回一个元组 (grid_nfeat, mesh_nfeat)。
            grid_nfeat: (N_grid, C_src_out)，更新后的 Grid 节点特征。
            mesh_nfeat: (N_mesh, C_dst_out)，更新后的 Mesh 节点特征。

    Example:
        >>> # 假设 Grid 有 1000 个节点，Mesh 有 200 个节点，边数为 3000
        >>> encoder = MeshGraphEncoder(
        ...     input_dim_src_nodes=64,
        ...     input_dim_dst_nodes=64,
        ...     input_dim_edges=32,
        ...     output_dim_src_nodes=64,
        ...     output_dim_dst_nodes=64,
        ...     output_dim_edges=32,
        ...     hidden_dim=128
        ... )
        >>> g2m_efeat = torch.randn(3000, 32)
        >>> grid_nfeat = torch.randn(1000, 64)
        >>> mesh_nfeat = torch.randn(200, 64)
        >>> # graph 为预定义的 Grid-to-Mesh 二部图对象
        >>> grid_out, mesh_out = encoder(g2m_efeat, grid_nfeat, mesh_nfeat, graph)
        >>> grid_out.shape
        torch.Size([1000, 64])
        >>> mesh_out.shape
        torch.Size([200, 64])
    """

    def __init__(
        self,
        aggregation: str = "sum",
        input_dim_src_nodes: int = 512,
        input_dim_dst_nodes: int = 512,
        input_dim_edges: int = 512,
        output_dim_src_nodes: int = 512,
        output_dim_dst_nodes: int = 512,
        output_dim_edges: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: int = nn.SiLU(),
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

        # src node MLP
        self.src_node_mlp = MeshGraphMLP(
            input_dim=input_dim_src_nodes,
            output_dim=output_dim_src_nodes,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

        # dst node MLP
        self.dst_node_mlp = MeshGraphMLP(
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
        g2m_efeat: Tensor,
        grid_nfeat: Tensor,
        mesh_nfeat: Tensor,
        graph: Union[DGLGraph, CuGraphCSC],
    ) -> Tuple[Tensor, Tensor]:
        # update edge features by concatenating node features (both mesh and grid) and existing edge featues
        # (or applying the concat trick instead)
        efeat = self.edge_mlp(g2m_efeat, (grid_nfeat, mesh_nfeat), graph)
        # aggregate messages (edge features) to obtain updated node features
        cat_feat = aggregate_and_concat(efeat, mesh_nfeat, graph, self.aggregation)
        # update src, dst node features + residual connections
        mesh_nfeat = mesh_nfeat + self.dst_node_mlp(cat_feat)
        grid_nfeat = grid_nfeat + self.src_node_mlp(grid_nfeat)
        return grid_nfeat, mesh_nfeat
