from typing import Tuple, Union

import torch
import torch.nn as nn
from dgl import DGLGraph
from torch import Tensor

# --- 引入模块化组件 ---
from onescience.modules import OneMlp
from onescience.modules.utils.gnnlayer_utils import CuGraphCSC, aggregate_and_concat

class MeshGraphEncoder(nn.Module):
    """
    用于 GraphCast 或 MeshGraphNet 等模型中的编码器模块 (Mesh Graph Encoder)。

    该模块作用于连接规则网格（Grid，代表输入物理域）和多尺度网格（Mesh，代表隐空间计算域）
    的二部图（Bipartite Graph）。它负责将信息从输入网格（源节点）编码并传递到隐空间网格（目标节点）。

    计算流程包括：
        1. 边特征更新: 利用 MeshGraphEdgeMLP 结合当前的边特征、Grid 节点特征和 Mesh 节点特征，
           来计算并更新 Grid-to-Mesh 的边特征。
        2. 消息聚合: 将更新后的边特征聚合到目标 Mesh 节点上，并与 Mesh 节点原始特征进行拼接。
        3. 节点特征更新:
            - Mesh 节点: 拼接后的特征通过 MLP 处理，并与原始特征进行残差相加。
            - Grid 节点: Grid 特征自身通过一个独立的 MLP 映射更新，同样应用残差连接。

    Args:
        aggregation (str, optional): 消息聚合方法，可选 "sum" 或 "mean"。默认值: "sum"。
        input_dim_src_nodes (int, optional): 输入源节点（Grid）特征的维度。默认值: 512。
        input_dim_dst_nodes (int, optional): 输入目标节点（Mesh）特征的维度。默认值: 512。
        input_dim_edges (int, optional): 输入二部图边特征的维度。默认值: 512。
        output_dim_src_nodes (int, optional): 输出源节点（Grid）特征的维度。默认值: 512。
        output_dim_dst_nodes (int, optional): 输出目标节点（Mesh）特征的维度。默认值: 512。
        output_dim_edges (int, optional): 输出边特征的维度。默认值: 512。
        hidden_dim (int, optional): MLP 隐藏层的神经元数量。默认值: 512。
        hidden_layers (int, optional): MLP 隐藏层的层数。默认值: 1。
        activation_fn (nn.Module, optional): 激活函数实例。默认值: nn.SiLU()。
        norm_type (str, optional): 归一化类型 ("LayerNorm" 或 "TELayerNorm")。默认值: "LayerNorm"。
        do_concat_trick (bool, optional): 是否使用特征拼接优化技巧 (使用 Sum 替代显式 Concat) 以节省显存。默认值: False。
        recompute_activation (bool, optional): 是否启用激活重计算机制以节省显存。默认值: False。

    形状:
        输入 g2m_efeat: (E, input_dim_edges)，Grid-to-Mesh 边的特征，E 为边数。
        输入 grid_nfeat: (N_grid, input_dim_src_nodes)，Grid 节点的特征。
        输入 mesh_nfeat: (N_mesh, input_dim_dst_nodes)，Mesh 节点的特征。
        输入 graph: DGLGraph 或 CuGraphCSC 对象，表示 Grid 到 Mesh 的有向图。
        输出: 返回一个元组 (grid_nfeat_out, mesh_nfeat_out)。
            - grid_nfeat_out: (N_grid, output_dim_src_nodes)
            - mesh_nfeat_out: (N_mesh, output_dim_dst_nodes)

    Example:
        >>> encoder = MeshGraphEncoder(
        ...     input_dim_src_nodes=64, input_dim_dst_nodes=64, input_dim_edges=32,
        ...     output_dim_src_nodes=64, output_dim_dst_nodes=64, output_dim_edges=32,
        ...     hidden_dim=128
        ... )
        >>> g2m_efeat = torch.randn(3000, 32)
        >>> grid_nfeat = torch.randn(1000, 64)
        >>> mesh_nfeat = torch.randn(200, 64)
        >>> # graph 为预定义的 Grid-to-Mesh 图对象
        >>> grid_out, mesh_out = encoder(g2m_efeat, grid_nfeat, mesh_nfeat, graph)
        >>> print(grid_out.shape)
        torch.Size([1000, 64])
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
        activation_fn: nn.Module = nn.SiLU(), # 修正原代码中 int 的 type hint 错误
        norm_type: str = "LayerNorm",
        do_concat_trick: bool = False,
        recompute_activation: bool = False,
    ):
        super().__init__()
        self.aggregation = aggregation

        edge_mlp_style = "MeshGraphEdgeMLPSum" if do_concat_trick else "MeshGraphEdgeMLPConcat"
        
        self.edge_mlp = OneMlp(
            style=edge_mlp_style,
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

        self.src_node_mlp = OneMlp(
            style="MeshGraphMLP",
            input_dim=input_dim_src_nodes,
            output_dim=output_dim_src_nodes,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

        self.dst_node_mlp = OneMlp(
            style="MeshGraphMLP",
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
        
        efeat = self.edge_mlp(g2m_efeat, (grid_nfeat, mesh_nfeat), graph)
        
        cat_feat = aggregate_and_concat(efeat, mesh_nfeat, graph, self.aggregation)
        
        mesh_nfeat = mesh_nfeat + self.dst_node_mlp(cat_feat)
        grid_nfeat = grid_nfeat + self.src_node_mlp(grid_nfeat)
        
        return grid_nfeat, mesh_nfeat