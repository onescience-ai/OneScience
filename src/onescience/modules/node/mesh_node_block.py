from typing import Tuple, Union

import torch
import torch.nn as nn
from dgl import DGLGraph
from torch import Tensor

from onescience.modules.mlp.mesh_graph_mlp import MeshGraphMLP
from onescience.modules.utils.gnnlayer_utils import CuGraphCSC, aggregate_and_concat

class MeshNodeBlock(nn.Module):
    """
    用于 GraphCast 或 MeshGraphNet 等模型中的节点更新块 (Node Block)。

    该模块在网格 (Mesh) 表示的隐空间上运行，负责更新节点特征。
    其计算过程包括：
        1. 消息聚合: 根据指定的聚合方法（如求和或平均），将连接到节点的边特征聚合起来，作为接收到的“消息”。
        2. 特征拼接: 将聚合后的消息与节点自身的当前特征进行拼接。
        3. MLP 变换与残差连接: 拼接后的特征通过一个 MLP 进行变换，变换结果与原始节点特征相加（残差连接），得到更新后的节点特征。

    Args:
        aggregation (str, optional): 消息聚合方法，可选 "sum" 或 "mean"。默认值: "sum"。
        input_dim_nodes (int, optional): 输入节点特征的维度。默认值: 512。
        input_dim_edges (int, optional): 输入边特征的维度（即传入消息的维度）。默认值: 512。
        output_dim (int, optional): 输出节点特征的维度。为了支持残差连接，通常应与 input_dim_nodes 保持一致。默认值: 512。
        hidden_dim (int, optional): MLP 隐藏层的神经元数量。默认值: 512。
        hidden_layers (int, optional): MLP 中隐藏层的层数。默认值: 1。
        activation_fn (nn.Module, optional): 激活函数类型。默认值: nn.SiLU()。
        norm_type (str, optional): 归一化类型 ("LayerNorm" 或 "TELayerNorm")。默认值: "LayerNorm"。
        recompute_activation (bool, optional): 是否启用激活重计算以节省显存。默认值: False。

    形状:
        输入 efeat: (E, C_edge)，其中 E 是边的数量。
        输入 nfeat: (N, C_node_in)，其中 N 是节点的数量。
        输入 graph: DGLGraph 或 CuGraphCSC 对象，定义图的拓扑结构。
        输出: 返回一个元组 (efeat, new_nfeat)。
            efeat: (E, C_edge)，原样返回的边特征（该模块不更新边）。
            new_nfeat: (N, C_node_out)，更新后的节点特征。

    Example:
        >>> # 假设有 100 个节点，500 条边
        >>> node_block = MeshNodeBlock(
        ...     aggregation="mean",
        ...     input_dim_nodes=64,
        ...     input_dim_edges=32,
        ...     output_dim=64,
        ...     hidden_dim=128
        ... )
        >>> efeat = torch.randn(500, 32)
        >>> nfeat = torch.randn(100, 64)
        >>> # graph 为图结构对象
        >>> _, new_nfeat = node_block(efeat, nfeat, graph)
        >>> new_nfeat.shape
        torch.Size([100, 64])
    """

    def __init__(
        self,
        aggregation: str = "sum",
        input_dim_nodes: int = 512,
        input_dim_edges: int = 512,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        recompute_activation: bool = False,
    ):
        super().__init__()
        self.aggregation = aggregation

        self.node_mlp = MeshGraphMLP(
            input_dim=input_dim_nodes + input_dim_edges,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

    @torch.jit.ignore()
    def forward(
        self,
        efeat: Tensor,
        nfeat: Tensor,
        graph: Union[DGLGraph, CuGraphCSC],
    ) -> Tuple[Tensor, Tensor]:
        # update edge features
        cat_feat = aggregate_and_concat(efeat, nfeat, graph, self.aggregation)
        # update node features + residual connection
        nfeat_new = self.node_mlp(cat_feat) + nfeat
        return efeat, nfeat_new
