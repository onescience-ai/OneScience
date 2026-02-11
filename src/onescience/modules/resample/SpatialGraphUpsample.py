import torch
import torch.nn as nn
import torch_geometric.nn as nng
import random
class SpatialGraphUpsample(nn.Module):

    """
    空间图上采样模块 (Spatial Graph Upsampling)。

    该模块对应于 CNN 中的 Unpooling 或 Upsample。
    它利用最近邻插值 (Nearest Neighbor Interpolation) 将低分辨率图的特征映射回高分辨率图的结构中。
    对于高分辨率图中的每个节点，寻找其在低分辨率图中空间距离最近的节点，并复制其特征。

    Args:
        无参数 (无状态模块)。

    形状:
        输入 x_down: (M, C)，低分辨率特征。
        输入 pos_down: (M, D)，低分辨率坐标。
        输入 pos_up: (N, D)，高分辨率坐标 (目标位置)。
        输出 x_up: (N, C)，上采样后的特征。

    Example:
        >>> upsample = SpatialGraphUpsample()
        >>> x_down = torch.randn(50, 32)   # 50 个点
        >>> pos_down = torch.randn(50, 2)
        >>> pos_up = torch.randn(100, 2)   # 恢复到 100 个点
        >>> x_up = upsample(x_down, pos_down, pos_up)
        >>> print(x_up.shape)
        torch.Size([100, 32])
    """
    def __init__(self):
        super().__init__()

    def forward(self, x_down, pos_down, pos_up):
        # cluster[i] 是 pos_up[i] 在 pos_down 中最近邻的索引
        cluster = nng.nearest(pos_up, pos_down)
        x_up = x_down[cluster]
        return x_up