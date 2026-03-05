import torch
import torch.nn as nn
from onescience.modules.mlp import StandardMLP as MLP

class RNNClusterPooling(nn.Module):
    """
        基于 RNN 的图池化模块。

        该模块将细粒度的节点特征聚合为粗粒度的簇（Cluster）特征。
        它首先根据簇分配索引收集节点特征，然后利用 GRU 处理每个簇内的节点序列，最终生成簇级别的特征表示。
        这种方法常用于从网格（Mesh）到潜在图（Latent Graph）的映射。

        Args:
            w_size (int): 输出簇特征的维度（Latent dimension）。
            pos_length (int): 位置编码长度，用于推断输入维度。

        形状:
            输入 V: (B, N, 128)，节点特征。
            输入 clusters: (B, K, C_max)，簇分配索引，表示每个簇包含的节点索引。
            输入 positional_encoding: (B, N, P)，节点位置编码。
            输入 cluster_mask: (B, K, C_max)，簇掩码，指示有效的节点。
            输出 W: (B, K, w_size)，池化后的簇特征。

        Example:
            >>> pool = RNNClusterPooling(w_size=512, pos_length=7)
            >>> # W = pool(V, clusters, pos_enc, mask)
            >>> # W.shape -> torch.Size([1, K, 512])
    """
    def __init__(self, w_size, pos_length):
        super(RNNClusterPooling, self).__init__()
        input_size = 128 + pos_length * 8

        self.rnn_pooling = nn.GRU(
            input_size=input_size, hidden_size=w_size, batch_first=True
        )
        self.linear_rnn = MLP(
            input_dim=w_size, 
            hidden_dims=[w_size], # n_hidden=1, hidden_size=w_size
            output_dim=w_size, 
            activation='relu',
            norm_layer=None
        )

    def forward(self, V, clusters, positional_encoding, cluster_mask):
        B, K, C_max = clusters.shape

        # 收集簇内节点的位置编码
        pos_gather_idx = clusters.reshape(B, -1, 1).repeat(1, 1, positional_encoding.shape[-1])
        pos_by_cluster = torch.gather(positional_encoding, -2, pos_gather_idx)
        pos_features = pos_by_cluster.reshape(B, K, C_max, -1)

        # 收集簇内节点的特征 V
        v_gather_idx = clusters.reshape(B, -1, 1).repeat(1, 1, V.shape[-1])
        V_by_cluster = torch.gather(V, -2, v_gather_idx)
        V_by_cluster = V_by_cluster.reshape(B, K, C_max, -1)

        # 拼接并输入 RNN
        inpt_by_cluster = torch.cat([V_by_cluster, pos_features], dim=-1)
        output, h = self.rnn_pooling(inpt_by_cluster.reshape(B * K, C_max, -1))

        # 获取 RNN 序列中最后一个有效节点的输出
        indices = (cluster_mask.sum(-1).long() - 1).reshape(B * K)
        indices[indices == -1] = output.shape[-2] - 1
        
        # 提取对应索引的输出
        w = torch.gather(
            output,
            1,
            indices.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, output.shape[-1]),
        )

        # 线性投影
        w = self.linear_rnn(w)
        W = w.reshape(B, K, -1)

        return W