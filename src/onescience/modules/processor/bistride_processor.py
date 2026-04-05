from typing import Optional

import torch
import torch.nn as nn
from onescience.modules import OneMlp

class BistrideGraphMessagePassing(nn.Module):
    """
    双步图消息传递网络 (Bistride Graph Message Passing, BSGMP)。

    这是一个用于分层图处理的核心模块，类似于图神经网络中的 U-Net 结构。
    它通过多层级的下采样（Pooling）和上采样（Unpooling）操作，在不同分辨率的图上进行特征提取和消息传递。

    该模块包含三个主要阶段：
    1. **Down Pass (编码路径)**: 逐层对图进行粗化（Pooling），并在每一层进行图消息传递 (GMP)。
    2. **Bottom Pass (瓶颈层)**: 在最粗糙的图层级上进行深度的图消息传递。
    3. **Up Pass (解码路径)**: 逐层恢复图的分辨率（Unpooling），并通过跳跃连接融合编码路径的特征。

    Args:
        unet_depth (int): U-Net 的深度（不包括最底层）。例如 depth=3 表示有 3 次下采样和 3 次上采样。
        latent_dim (int): 图节点和边的潜在特征维度。
        hidden_layer (int): MLP 内部隐藏层的数量。
        pos_dim (int): 物理坐标的维度（例如 2D 对应 2，3D 对应 3）。

    形状:
        输入 h: (B, N, latent_dim) 或 (N, latent_dim)，初始节点特征。
        输入 m_ids: list[Tensor]，每一层下采样的索引列表。
        输入 m_gs: list[Tensor]，每一层的图连接结构（边索引）。m_gs[0] 是原始图，后续是粗化图。
        输入 pos: (B, N, pos_dim) 或 (N, pos_dim)，初始节点坐标。
        输出: (B, N, latent_dim) 或 (N, latent_dim)，更新后的节点特征，形状与输入 h 相同。

    Example:
        >>> bsgmp = BistrideGraphMessagePassing(unet_depth=2, latent_dim=64, hidden_layer=2, pos_dim=3)
        >>> # forward 调用需要传入多尺度的图结构数据 m_ids 和 m_gs
        >>> # h_out = bsgmp(h, m_ids, m_gs, pos)
    """

    def __init__(self, unet_depth, latent_dim, hidden_layer, pos_dim):
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
        down_outs = []
        down_ps = []
        cts = []

        w = pos.new_ones((pos.shape[-2], 1))

        # Down pass
        for i in range(self.unet_depth):
            h = self.down_gmps[i](h, m_gs[i], pos)
            down_outs.append(h)
            down_ps.append(pos)

            ew, w = self.edge_conv.cal_ew(w, m_gs[i])
            h = self.edge_conv(h, m_gs[i], ew)
            pos = self.edge_conv(pos, m_gs[i], ew)
            cts.append(ew)

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
            h = self.edge_conv(h, g, cts[depth_idx], aggragating=False)
            h = self.up_gmps[i](h, g, down_ps[depth_idx])
            h = h.add(down_outs[depth_idx])

        return h


class GraphMessagePassing(nn.Module):
    """
    图消息传递块 (Graph Message Passing Block)。

    该模块用于在图结构上执行一次完整的消息传递 (Message Passing)。
    它首先根据节点特征和节点间的物理距离 (方向和欧氏范数) 构建边特征，
    使用 mlp_edge 提取边信息并聚合到目标节点上，最后使用 mlp_node 更新节点特征。

    Args:
        latent_dim (int): 节点和边的潜空间特征维度。
        hidden_layer (int): 内部 MLP 的隐藏层层数。
        pos_dim (int): 位置坐标的维度 (例如 2D=2, 3D=3)。

    形状:
        输入 x: (B, N, C) 或 (N, C)，节点特征。
        输入 g: (2, E)，图的边连接索引 (COO 格式)。
        输入 pos: (B, N, pos_dim) 或 (N, pos_dim)，节点坐标。
        输出: (B, N, C) 或 (N, C)，更新后的节点特征。

    Example:
        >>> gmp = GraphMessagePassing(latent_dim=64, hidden_layer=2, pos_dim=3)
        >>> x = torch.randn(100, 64)      # 100个节点
        >>> pos = torch.randn(100, 3)     # 100个3D坐标
        >>> g = torch.randint(0, 100, (2, 300)) # 300条边
        >>> out = gmp(x, g, pos)
        >>> print(out.shape)
        torch.Size([100, 64])
    """

    def __init__(self, latent_dim, hidden_layer, pos_dim):
        super().__init__()
        self.pos_dim = pos_dim
        
        self.mlp_node = OneMlp(
            style="MeshGraphMLP",
            input_dim=2 * latent_dim,
            output_dim=latent_dim,
            hidden_dim=[latent_dim] * hidden_layer
        )
        
        edge_info_in_len = 2 * latent_dim + pos_dim + 1
        self.mlp_edge = OneMlp(
            style="MeshGraphMLP",
            input_dim=edge_info_in_len,
            output_dim=latent_dim,
            hidden_dim=[latent_dim] * hidden_layer
        )

    def forward(self, x, g, pos):
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
            raise ValueError(f"Only implemented for dim 2 and 3, got {pos.shape}")

        dir = pi - pj
        norm = torch.norm(dir, dim=-1, keepdim=True)
        fiber = torch.cat([dir, norm], dim=-1)

        if len(x.shape) == 3 and len(pos.shape) == 2:
            tmp = torch.cat([fiber.unsqueeze(0).repeat(B, 1, 1), x_i, x_j], dim=-1)
        else:
            tmp = torch.cat([fiber, x_i, x_j], dim=-1)
            
        edge_embedding = self.mlp_edge(tmp)
        aggr_out = scatter_sum(edge_embedding, j, dim=-2, dim_size=x.shape[-2])

        tmp = torch.cat([x, aggr_out], dim=-1)
        return self.mlp_node(tmp) + x


class WeightedEdgeConv(nn.Module):
    """
    加权边卷积层 (Weighted Edge Convolution)。

    用于在 U-Net 的池化 (Down) 和反池化 (Up) 阶段对特征进行跨层传递。
    它可以根据预先计算好的边权重 (Edge Weights)，将邻居节点的信息加权聚合到中心节点。

    Args:
        None

    形状:
        输入 x: (B, N, C) 或 (N, C)。
        输入 g: (2, E)，边索引。
        输入 ew: (E,)，每条边的权重。
        输入 aggragating: bool，True为向下聚合，False为向上分配。
        输出: (B, N, C) 或 (N, C)。
    """

    def __init__(self, *args):
        super(WeightedEdgeConv, self).__init__()

    def forward(self, x, g, ew, aggragating=True):
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
        """计算边权重，用于前向传播中的加权操作。"""
        deg = degree(g[0], dtype=torch.float, num_nodes=w.shape[0])
        normed_w = w.squeeze(-1) / deg
        i, j = g[0], g[1]
        w_to_send = normed_w[i]
        eps = 1e-12
        aggr_w = scatter_sum(w_to_send, j, dim=-1, dim_size=normed_w.size(0)) + eps
        ec = w_to_send / aggr_w[j]

        return ec, aggr_w


class Unpool(nn.Module):
    """
    图反池化层 (Unpooling Layer)。

    用于在 BSGMP 的 Up Pass 中，将粗糙层级的图节点特征映射回精细层级的对应节点上，
    未映射的节点特征将被填充为零，等待后续的残差连接和边卷积补充信息。

    Args:
        None

    形状:
        输入 h: (B, N, C) 或 (N, C)，粗糙层特征。
        输入 pre_node_num: int，目标精细层的节点总数。
        输入 idx: (N,) 或 (B, N)，粗糙节点在精细图中的绝对索引。
        输出: (B, pre_node_num, C) 或 (pre_node_num, C)。
    """

    def __init__(self, *args):
        super(Unpool, self).__init__()

    def forward(self, h, pre_node_num, idx):
        if len(h.shape) == 2:
            new_h = h.new_zeros([pre_node_num, h.shape[-1]])
            new_h[idx] = h
        elif len(h.shape) == 3:
            new_h = h.new_zeros([h.shape[0], pre_node_num, h.shape[-1]])
            new_h[:, idx] = h

        return new_h


# =====================================================================
# 底层工具函数 (Utility Functions)
# =====================================================================

def degree(
    index: torch.Tensor,
    num_nodes: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    功能：计算一维索引张量中每个节点的（无权）度数 (Degree)。
    即将边索引张量中指向同一个节点的次数进行累加统计。
    """
    N = torch.max(index) + 1 if num_nodes is None else num_nodes
    N = int(N)
    out = torch.zeros((N,), dtype=dtype, device=index.device)
    one = torch.ones((index.size(0),), dtype=out.dtype, device=out.device)
    return out.scatter_add_(0, index, one)


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    """
    功能：广播张量形状。
    将 src 张量沿指定维度 dim 广播扩展，使其形状与 other 张量对齐，以便后续的逐元素操作。
    """
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
    """
    功能：执行 scatter sum (分散求和) 聚合操作。
    将 src 中的数据沿着指定维度 dim，根据 index 提供的目标索引累加到输出张量中。
    这是图神经网络中将“边特征”聚合到“节点”的最核心底层操作。
    """
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