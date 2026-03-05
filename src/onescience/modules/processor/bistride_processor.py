from typing import Optional

import torch
import torch.nn as nn

from onescience.modules.mlp.mesh_graph_mlp import MeshGraphMLP


class BistrideGraphMessagePassing(nn.Module):
    """
        双步图消息传递网络 (Bistride Graph Message Passing, BSGMP)。

        这是一个用于分层图处理的核心模块，类似于图神经网络中的 U-Net 结构。
        它通过多层级的下采样（Pooling）和上采样（Unpooling）操作，在不同分辨率的图上进行特征提取和消息传递。

        该模块包含三个主要阶段：
        1. **Down Pass (编码路径)**: 逐层对图进行粗化（Pooling），并在每一层进行图消息传递 (GMP)。
        2. **Bottom Pass (瓶颈层)**: 在最粗糙的图层级上进行深度的图消息传递。
        3. **Up Pass (解码路径)**: 逐层恢复图的分辨率（Unpooling），并通过跳跃连接（Skip Connection）融合编码路径的特征，进行图消息传递。

        Args:
            unet_depth (int): U-Net 的深度（不包括最底层）。例如 depth=3 表示有 3 次下采样和 3 次上采样。
            latent_dim (int): 图节点和边的潜在特征维度。
            hidden_layer (int): MLP 内部隐藏层的数量。
            pos_dim (int): 物理坐标的维度（例如 2D 或 3D）。

        形状:
            输入 h: (B, N, latent_dim) 或 (N, latent_dim)，初始节点特征。
            输入 m_ids: list[Tensor]，每一层下采样的索引列表。
            输入 m_gs: list[Tensor]，每一层的图连接结构（边索引）。m_gs[0] 是原始图，后续是粗化图。
            输入 pos: (B, N, pos_dim) 或 (N, pos_dim)，初始节点坐标。
            输出: (B, N, latent_dim) 或 (N, latent_dim)，更新后的节点特征，形状与输入 h 相同。

        Example:
            >>> # 假设 latent_dim=128, pos_dim=3
            >>> bsgmp = BistrideGraphMessagePassing(unet_depth=3, latent_dim=128, hidden_layer=2, pos_dim=3)
            >>> # forward 调用需要传入多尺度的图结构数据 m_ids 和 m_gs
            >>> # h_out = bsgmp(h, m_ids, m_gs, pos)
    """

    def __init__(self, unet_depth, latent_dim, hidden_layer, pos_dim):
        """
        Initializes the BSGMP network.

        Parameters
        ----------
        unet_depth : int
            UNet depth in the network, excluding top level.
        latent_dim : int
            Latent dimension for the graph nodes and edges.
        hidden_layer : int
            Number of hidden layers in the MLPs.
        pos_dim : int
            Dimension of the physical position (in Euclidean space).
        """
        super().__init__()

        self.bottom_gmp = GraphMessagePassing(latent_dim, hidden_layer, pos_dim)
        self.down_gmps = nn.ModuleList()
        self.up_gmps = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.unet_depth = unet_depth
        self.edge_conv = WeightedEdgeConv()
        for _ in range(self.unet_depth):
            self.down_gmps.append(
                GraphMessagePassing(latent_dim, hidden_layer, pos_dim)
            )
            self.up_gmps.append(GraphMessagePassing(latent_dim, hidden_layer, pos_dim))
            self.unpools.append(Unpool())

    def forward(self, h, m_ids, m_gs, pos):
        down_outs = []  # to store output features at each level during down pass
        down_ps = []  # to store positional information at each level during down pass
        cts = []  # to store edge weights for convolution at each level

        w = pos.new_ones((pos.shape[-2], 1))  # Initialize weights

        # Down pass
        for i in range(self.unet_depth):
            h = self.down_gmps[i](h, m_gs[i], pos)
            down_outs.append(h)
            down_ps.append(pos)

            # Calculate edge weights
            ew, w = self.edge_conv.cal_ew(w, m_gs[i])
            h = self.edge_conv(h, m_gs[i], ew)
            pos = self.edge_conv(pos, m_gs[i], ew)
            cts.append(ew)

            # Pooling
            if len(h.shape) == 3:
                h = h[:, m_ids[i]]
            elif len(h.shape) == 2:
                h = h[m_ids[i]]

            if len(pos.shape) == 3:
                pos = pos[:, m_ids[i]]
            elif len(pos.shape) == 2:
                pos = pos[m_ids[i]]

            w = w[m_ids[i]]

        # Bottom pass
        h = self.bottom_gmp(h, m_gs[self.unet_depth], pos)

        # Up pass
        for i in range(self.unet_depth):
            depth_idx = self.unet_depth - i - 1
            g, idx = m_gs[depth_idx], m_ids[depth_idx]
            h = self.unpools[i](h, down_outs[depth_idx].shape[-2], idx)
            # aggregate is False as we are returning the information to previous out degrees.
            h = self.edge_conv(h, g, cts[depth_idx], aggragating=False)
            h = self.up_gmps[i](h, g, down_ps[depth_idx])
            h = h.add(down_outs[depth_idx])

        return h


class GraphMessagePassing(nn.Module):
    """Graph Message Passing (GMP) block."""

    def __init__(self, latent_dim, hidden_layer, pos_dim):
        """
        Initialize the GMP block.

        Parameters
        ----------
        latent_dim : int
            Dimension of the latent space.
        hidden_layer : int
            Number of hidden layers.
        pos_dim : int
            Dimension of the positional encoding.
        """
        super().__init__()
        self.mlp_node = MeshGraphMLP(
            2 * latent_dim, latent_dim, latent_dim, hidden_layer
        )
        edge_info_in_len = 2 * latent_dim + pos_dim + 1
        self.mlp_edge = MeshGraphMLP(
            edge_info_in_len, latent_dim, latent_dim, hidden_layer
        )
        self.pos_dim = pos_dim

    def forward(self, x, g, pos):
        """
        Forward pass for GMP block.

        Parameters
        ----------
        x : torch.Tensor
            Input node features of shape [B, N, C] or [N, C].
        g : torch.Tensor
            Graph connectivity (edges) of shape [2, E].
        pos : torch.Tensor
            Node positional information of shape [B, N, pos_dim] or [N, pos_dim].

        Returns
        -------
        torch.Tensor
            Updated node features.
        """
        i, j = g[0], g[1]

        if len(x.shape) == 3:
            B, _, _ = x.shape
            x_i, x_j = x[:, i], x[:, j]
        elif len(x.shape) == 2:
            x_i, x_j = x[i], x[j]
        else:
            raise ValueError(f"Only implemented for dim 2 and 3, got {x.shape}")

        if len(pos.shape) == 3:
            pi, pj = pos[:, i], pos[:, j]
        elif len(pos.shape) == 2:
            pi, pj = pos[i], pos[j]
        else:
            raise ValueError(f"Only implemented for dim 2 and 3, got {x.shape}")

        # Here is the biggest difference between BSMS's GMP and that of MeshGraphNet.
        # In MGN, the edge information is:
        # 1)initialized using fiber=(dir, norm)
        # 2)then it follows the MP times of MLP_edge, using the same graph connectivity.
        # In BSMS's GMP, since there is only 1 time of MP per layer
        # we dive into a deeper layer, i.e. the original edges are gone
        # it then does not make any sense to use 2) above
        # so we just use the fiber to cat with the in/out node features
        dir = pi - pj  # (B, N, pos_dim) or (N, pos_dim)
        norm = torch.norm(dir, dim=-1, keepdim=True)  # (B, N, 1) or (N, 1)
        fiber = torch.cat([dir, norm], dim=-1)  # (B, N, pos_dim+1) or (N, pos_dim+1)
        # below is the cat between fiber and node latent features
        if len(x.shape) == 3 and len(pos.shape) == 2:
            tmp = torch.cat([fiber.unsqueeze(0).repeat(B, 1, 1), x_i, x_j], dim=-1)
        else:
            tmp = torch.cat([fiber, x_i, x_j], dim=-1)
        # get the information flow on the edge
        edge_embedding = self.mlp_edge(tmp)
        # sum the edge information to the in node
        aggr_out = scatter_sum(edge_embedding, j, dim=-2, dim_size=x.shape[-2])

        # MLP take input as the cat between x and the aggregated edge information flow
        tmp = torch.cat([x, aggr_out], dim=-1)
        return self.mlp_node(tmp) + x


class WeightedEdgeConv(nn.Module):
    """Weighted Edge Convolution layer for transition between layers."""

    def __init__(self, *args):
        super(WeightedEdgeConv, self).__init__()

    def forward(self, x, g, ew, aggragating=True):
        """
        Forward pass for WeightedEdgeConv layer.

        Parameters
        ----------
        x : torch.Tensor
            Input node features of shape [B, N, C] or [N, C].
        g : torch.Tensor
            Graph connectivity (edges) of shape [2, E].
        ew : torch.Tensor
            Edge weights for convolution of shape [E].
        aggragating : bool, optional
            If True, aggregate messages (used in down pass); if False, return messages (used in up pass).

        Returns
        -------
        torch.Tensor
            Aggregated or scattered node features.
        """
        i, j = g[0], g[1]

        if len(x.shape) == 3:
            weighted_info = x[:, i] if aggragating else x[:, j]
        elif len(x.shape) == 2:
            weighted_info = x[i] if aggragating else x[j]
        else:
            raise NotImplementedError("Only implemented for dim 2 and 3")

        weighted_info *= ew.unsqueeze(-1)
        target_index = j if aggragating else i
        aggr_out = scatter_sum(
            weighted_info, target_index, dim=-2, dim_size=x.shape[-2]
        )

        return aggr_out

    @torch.no_grad()
    def cal_ew(self, w, g):
        """
        Calculate the edge weights for later use in forward.

        Parameters
        ----------
        w : torch.Tensor
            Node weights of shape [N, 1].
        g : torch.Tensor
            Graph connectivity (edges) of shape [2, E].

        Returns
        -------
        tuple
            Edge weights for convolution and aggregated node weights (used for iteratively calculating this in the next layer).
        """
        deg = degree(g[0], dtype=torch.float, num_nodes=w.shape[0])
        normed_w = w.squeeze(-1) / deg
        i, j = g[0], g[1]
        w_to_send = normed_w[i]
        eps = 1e-12
        aggr_w = scatter_sum(w_to_send, j, dim=-1, dim_size=normed_w.size(0)) + eps
        ec = w_to_send / aggr_w[j]

        return ec, aggr_w


class Unpool(nn.Module):
    """Unpooling layer for graph neural networks."""

    def __init__(self, *args):
        super(Unpool, self).__init__()

    def forward(self, h, pre_node_num, idx):
        """
        Forward pass for the unpooling layer.

        Parameters
        ----------
        h : torch.Tensor
            Node features of shape [N, C] or [B, N, C].
        pre_node_num : int
            Number of nodes in the previous upper layer.
        idx : torch.Tensor
            Relative indices (in the previous upper layer) for unpooling of shape [N] or [B, N].

        Returns
        -------
        torch.Tensor
            Unpooled node features of shape [pre_node_num, C] or [B, pre_node_num, C].
        """
        if len(h.shape) == 2:
            new_h = h.new_zeros([pre_node_num, h.shape[-1]])
            new_h[idx] = h
        elif len(h.shape) == 3:
            new_h = h.new_zeros([h.shape[0], pre_node_num, h.shape[-1]])
            new_h[:, idx] = h

        return new_h


def degree(
    index: torch.Tensor,
    num_nodes: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    r"""Computes the (unweighted) degree of a given one-dimensional index tensor.

    Args:
        index (LongTensor): Index tensor.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
        dtype (:obj:`torch.dtype`, optional): The desired data type of the
            returned tensor.

    :rtype: :class:`Tensor`

    Example:
        >>> row = torch.tensor([0, 1, 0, 2, 0])
        >>> degree(row, dtype=torch.long)
        tensor([3, 1, 1])
    """
    N = torch.max(index) + 1
    N = int(N)
    out = torch.zeros((N,), dtype=dtype, device=index.device)
    one = torch.ones((index.size(0),), dtype=out.dtype, device=out.device)
    return out.scatter_add_(0, index, one)


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)
