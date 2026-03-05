from typing import Tuple
import torch.nn as nn
from torch import Tensor

from onescience.modules.mlp.mesh_graph_mlp import MeshGraphMLP

class GraphCastEncoderEmbedder(nn.Module):
    """
    GraphCast 编码器嵌入层 (GraphCast Encoder Embedder)。

    该模块负责将 GraphCast 模型所需的四类输入特征映射到高维隐空间：
    1. 网格节点特征 (Grid Node Features)
    2. 多尺度网格节点特征 (Multi-mesh Node Features)
    3. 网格到多尺度网格的边特征 (Grid2Mesh Edge Features)
    4. 多尺度网格内部边特征 (Multi-mesh Edge Features)

    每个特征都通过一个独立的 MLP 进行嵌入。

    Args:
        input_dim_grid_nodes (int, optional): 网格节点特征的输入维度。默认值: 474。
        input_dim_mesh_nodes (int, optional): 多尺度网格节点特征的输入维度。默认值: 3。
        input_dim_edges (int, optional): 边特征的输入维度。默认值: 4。
        output_dim (int, optional): 嵌入后的特征维度 (Latent Dim)。默认值: 512。
        hidden_dim (int, optional): MLP 隐藏层神经元数量。默认值: 512。
        hidden_layers (int, optional): MLP 隐藏层层数。默认值: 1。
        activation_fn (nn.Module, optional): 激活函数类型。默认值: nn.SiLU()。
        norm_type (str, optional): 归一化类型。默认值: "LayerNorm"。
        recompute_activation (bool, optional): 是否在反向传播中重计算激活以节省显存。默认值: False。

    形状:
        输入 grid_nfeat: (N_grid, input_dim_grid_nodes)
        输入 mesh_nfeat: (N_mesh, input_dim_mesh_nodes)
        输入 g2m_efeat: (N_g2m, input_dim_edges)
        输入 mesh_efeat: (N_mesh_edges, input_dim_edges)
        输出: 返回一个包含四个张量的元组，形状分别为:
            - (N_grid, output_dim)
            - (N_mesh, output_dim)
            - (N_g2m, output_dim)
            - (N_mesh_edges, output_dim)

    Example:
        >>> embedder = GraphCastEncoderEmbedder(
        ...     input_dim_grid_nodes=474,
        ...     input_dim_mesh_nodes=3,
        ...     input_dim_edges=4,
        ...     output_dim=512
        ... )
        >>> grid_n = torch.randn(100, 474)
        >>> mesh_n = torch.randn(50, 3)
        >>> g2m_e = torch.randn(200, 4)
        >>> mesh_e = torch.randn(300, 4)
        >>> out_grid, out_mesh, out_g2m, out_mesh_e = embedder(grid_n, mesh_n, g2m_e, mesh_e)
        >>> print(out_grid.shape)
        torch.Size([100, 512])
    """

    def __init__(
        self,
        input_dim_grid_nodes: int = 474,
        input_dim_mesh_nodes: int = 3,
        input_dim_edges: int = 4,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        recompute_activation: bool = False,
    ):
        super().__init__()

        # MLP for grid node embedding
        self.grid_node_mlp = MeshGraphMLP(
            input_dim=input_dim_grid_nodes,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

        # MLP for mesh node embedding
        self.mesh_node_mlp = MeshGraphMLP(
            input_dim=input_dim_mesh_nodes,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

        # MLP for mesh edge embedding
        self.mesh_edge_mlp = MeshGraphMLP(
            input_dim=input_dim_edges,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

        # MLP for grid2mesh edge embedding
        self.grid2mesh_edge_mlp = MeshGraphMLP(
            input_dim=input_dim_edges,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

    def forward(self, x) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # Input node feature embedding
        grid_nfeat = x[0]
        mesh_nfeat = x[1]
        g2m_efeat = x[2]
        mesh_efeat = x[3]
        grid_nfeat = self.grid_node_mlp(grid_nfeat)
        mesh_nfeat = self.mesh_node_mlp(mesh_nfeat)
        # Input edge feature embedding
        g2m_efeat = self.grid2mesh_edge_mlp(g2m_efeat)
        mesh_efeat = self.mesh_edge_mlp(mesh_efeat)
        return grid_nfeat, mesh_nfeat, g2m_efeat, mesh_efeat


class GraphCastDecoderEmbedder(nn.Module):
    """
    GraphCast 解码器嵌入层 (GraphCast Decoder Embedder)。

    该模块用于将多尺度网格回到原始网格 (Mesh2Grid) 的边特征进行嵌入。
    这是 GraphCast 解码过程的第一步，用于处理从 latent mesh 传回 grid 的信息。

    Args:
        input_dim_edges (int, optional): 输入边特征的维度。默认值: 4。
        output_dim (int, optional): 嵌入后的特征维度。默认值: 512。
        hidden_dim (int, optional): MLP 隐藏层神经元数量。默认值: 512。
        hidden_layers (int, optional): MLP 隐藏层层数。默认值: 1。
        activation_fn (nn.Module, optional): 激活函数类型。默认值: nn.SiLU()。
        norm_type (str, optional): 归一化类型 ["TELayerNorm", "LayerNorm"]。默认值: "LayerNorm"。
        recompute_activation (bool, optional): 是否重计算激活。默认值: False。

    形状:
        输入 m2g_efeat: (N_m2g, input_dim_edges)，Mesh2Grid 的边特征。
        输出: (N_m2g, output_dim)，嵌入后的边特征。

    Example:
        >>> embedder = GraphCastDecoderEmbedder(input_dim_edges=4, output_dim=512)
        >>> m2g_edge = torch.randn(150, 4)
        >>> out = embedder(m2g_edge)
        >>> print(out.shape)
        torch.Size([150, 512])
    """

    def __init__(
        self,
        input_dim_edges: int = 4,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        recompute_activation: bool = False,
    ):
        super().__init__()

        # MLP for mesh2grid edge embedding
        self.mesh2grid_edge_mlp = MeshGraphMLP(
            input_dim=input_dim_edges,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

    def forward(
        self,
        m2g_efeat: Tensor,
    ) -> Tensor:
        m2g_efeat = self.mesh2grid_edge_mlp(m2g_efeat)
        return m2g_efeat