import torch
import torch.nn as nn
import torch_geometric.nn as nng
import random

class SpatialGraphDownsample(nn.Module):
    """
    空间图下采样模块 (Spatial Graph Downsampling)。

    该模块执行两个核心操作：
    1. **节点选择 (Pooling)**: 减少节点数量。支持 'random' (随机采样) 或 'topk' (基于投影分数的 TopKPooling)。
    2. **图拓扑重构 (Topology Reconstruction)**: 基于采样后节点的空间位置，利用半径图 (Radius Graph) 算法重新构建邻接关系。
    
    这模拟了 CNN 中的 Pooling 操作，但在非结构化几何数据（如点云、网格）上进行。

    Args:
        in_channels (int): 输入特征维度（仅当 pool_method='topk' 时需要，用于计算投影分数）。
        ratio (float, optional): 池化比率 (保留节点的比例)。默认值: 0.5。
        r (float, optional): 半径图构建时的半径阈值。默认值: 0.1。
        max_num_neighbors (int, optional): 每个节点的最大邻居数。默认值: 64。
        pool_method (str, optional): 池化方法，支持 'random' 或 'topk'。默认值: 'random'。

    形状:
        输入 x: (N, C)，节点特征。
        输入 pos: (N, D)，节点坐标 (通常 D=2 或 3)。
        输入 edge_index (可选): (2, E)，当使用 'topk' 时需要原始边信息。
        输出 x_pooled: (M, C)，下采样后的特征，其中 M = N * ratio。
        输出 pos_pooled: (M, D)，下采样后的坐标。
        输出 edge_index_pooled: (2, E_new)，重构后的邻接矩阵。
        输出 perm: (M,)，被选中节点的索引。

    Example:
        >>> # 假设有 100 个节点，特征维度 32，2D 坐标
        >>> downsample = SpatialGraphDownsample(in_channels=32, ratio=0.5, r=0.2, pool_method='random')
        >>> x = torch.randn(100, 32)
        >>> pos = torch.randn(100, 2)
        >>> x_pool, pos_pool, edge_index_pool, perm = downsample(x, pos)
        >>> print(x_pool.shape)
        torch.Size([50, 32])
    """
    def __init__(self, in_channels, ratio=0.5, r=0.1, max_num_neighbors=64, pool_method='random'):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.max_num_neighbors = max_num_neighbors
        self.pool_method = pool_method

        if self.pool_method == 'topk':
            self.scorer = nng.TopKPooling(in_channels, ratio=ratio, nonlinearity=torch.sigmoid)
        else:
            self.scorer = None

    def forward(self, x, pos, edge_index=None):
        num_nodes = x.size(0)

        if self.scorer is not None:
            # TopK Pooling: 需要 edge_index 来传播分数
            if edge_index is None:
                raise ValueError("edge_index is required for TopK pooling")
            # x, edge_index, edge_attr, batch, perm, score
            x_pooled, _, _, _, perm, _ = self.scorer(x, edge_index)
        else:
            # Random Pooling
            k = int((self.ratio * float(num_nodes)))
            # 保持 device 一致
            perm = torch.randperm(num_nodes, device=x.device)[:k]
            x_pooled = x[perm]

        pos_pooled = pos[perm]

        # 基于新的空间位置重构图结构
        edge_index_pooled = nng.radius_graph(
            x=pos_pooled,
            r=self.r,
            loop=True,
            max_num_neighbors=self.max_num_neighbors
        )

        return x_pooled, pos_pooled, edge_index_pooled, perm