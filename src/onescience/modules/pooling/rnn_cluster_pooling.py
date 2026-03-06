import torch
import torch.nn as nn
from onescience.modules import OneMlp

class RNNClusterPooling(nn.Module):
    """
    基于 RNN 的图池化模块 (RNN-based Graph Cluster Pooling)。

    该模块用于将细粒度的节点特征聚合为粗粒度的簇 (Cluster) 特征。
    它首先根据簇分配索引收集节点特征，然后将每个簇内的节点视为一个序列，利用 GRU 进行序列处理，
    并提取序列最后一个有效节点的隐状态作为簇的整体特征表示。
    这种方法常用于图神经网络中的层级池化，如从细粒度网格 (Mesh) 映射到粗粒度潜在图 (Latent Graph)。

    Args:
        w_size (int): 输出簇特征的特征维度 (Latent Dimension)。
        pos_length (int): 位置编码长度参数。注意：模块内部硬编码假设输入节点特征 V 的维度为 128，
                          位置编码的实际特征维度为 pos_length * 8。

    形状:
        输入 V: (B, N, 128)，节点特征。
        输入 clusters: (B, K, C_max)，簇分配索引。K 为簇的数量，C_max 为每个簇最大节点数。
        输入 positional_encoding: (B, N, P)，节点位置编码，其中 P = pos_length * 8。
        输入 cluster_mask: (B, K, C_max)，簇掩码，用于指示每个簇内哪些是有效节点。
        输出 W: (B, K, w_size)，池化后生成的簇特征。

    Example:
        >>> pool = RNNClusterPooling(w_size=256, pos_length=8)
        >>> V = torch.randn(2, 1000, 128)                  # B=2, N=1000
        >>> clusters = torch.randint(0, 1000, (2, 50, 20)) # K=50, C_max=20
        >>> pos_enc = torch.randn(2, 1000, 64)             # 8 * 8 = 64
        >>> mask = torch.ones(2, 50, 20)                   # 简化全为有效节点
        >>> W = pool(V, clusters, pos_enc, mask)
        >>> print(W.shape)
        torch.Size([2, 50, 256])
    """
    def __init__(self, w_size, pos_length):
        super(RNNClusterPooling, self).__init__()
        
        input_size = 128 + pos_length * 8

        self.rnn_pooling = nn.GRU(
            input_size=input_size, hidden_size=w_size, batch_first=True
        )
        
        self.linear_rnn = OneMlp(
            style="StandardMLP",
            input_dim=w_size,
            output_dim=w_size,
            hidden_dims=[w_size], 
            activation='relu',
            use_bias=True
        )

    def forward(self, V, clusters, positional_encoding, cluster_mask):
        B, K, C_max = clusters.shape

        pos_gather_idx = clusters.reshape(B, -1, 1).repeat(1, 1, positional_encoding.shape[-1])
        pos_by_cluster = torch.gather(positional_encoding, -2, pos_gather_idx)
        pos_features = pos_by_cluster.reshape(B, K, C_max, -1)

        v_gather_idx = clusters.reshape(B, -1, 1).repeat(1, 1, V.shape[-1])
        V_by_cluster = torch.gather(V, -2, v_gather_idx)
        V_by_cluster = V_by_cluster.reshape(B, K, C_max, -1)

        inpt_by_cluster = torch.cat([V_by_cluster, pos_features], dim=-1)
        output, h = self.rnn_pooling(inpt_by_cluster.reshape(B * K, C_max, -1))

        indices = (cluster_mask.sum(-1).long() - 1).reshape(B * K)
        indices[indices == -1] = output.shape[-2] - 1
        
        w = torch.gather(
            output,
            1,
            indices.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, output.shape[-1]),
        )

        w = self.linear_rnn(w)
        W = w.reshape(B, K, -1)

        return W