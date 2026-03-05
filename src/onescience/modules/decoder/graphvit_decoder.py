import torch
import torch.nn as nn
from onescience.modules.mlp import StandardMLP as MLP
from onescience.modules.layer.gnn_layer import GNNLayer

class GraphViTDecoder(nn.Module):
    """
        基于 GNN 的解码/检索模块。

        该模块的功能与池化相反，它利用粗粒度的簇特征（W）和细粒度的节点特征（V）来恢复或预测节点的物理状态更新。
        它通过 GNN 进行信息传播，将全局（簇）信息融合回局部（节点）图结构中。

        Args:
            w_size (int): 输入簇特征的维度。
            pos_length (int): 位置编码长度。
            state_size (int): 最终输出的状态维度（例如预测的速度增量）。

        形状:
            输入 W: (B, K, w_size)，簇特征。
            输入 V: (B, N, 128)，节点特征。
            输入 clusters: (B, K, C_max)，簇分配索引。
            输入 positional_encoding: (B, N, P)，位置编码。
            输入 edges: (B, M, 2)，细粒度图的边索引。
            输入 E: (B, M, 128)，细粒度图的边特征。
            输出 final_state: (B, N, state_size)，预测的节点状态更新量。

        Example:
            >>> decoder = GraphViTDecoder(w_size=512, pos_length=7, state_size=2)
            >>> # out = decoder(W, V, clusters, pos_enc, edges, E)
            >>> # out.shape -> torch.Size([1, N, 2])
    """
    def __init__(self, w_size, pos_length, state_size):
        super(GraphViTDecoder, self).__init__()
        pos_size = pos_length * 8
        node_size = w_size + 128 + pos_size
        
        self.gnn = GNNLayer(node_size=node_size, output_size=128)
        self.final_mlp = MLP(
            input_dim=128,
            hidden_dims=[128, 128], 
            output_dim=state_size,
            activation='tanh',      
            output_activation=None, 
            norm_layer=None
        )

    def forward(self, W, V, clusters, positional_encoding, edges, E):
        B, N, _ = V.shape
        K = clusters.shape[1]
        C_dim = W.shape[-1]

        W_expanded = W.unsqueeze(-2).repeat(1, 1, clusters.shape[-1], 1).view(B, -1, C_dim)
        
        cluster_indices_flat = clusters.reshape(B, -1, 1).repeat(1, 1, C_dim)
        
        W_nodes = torch.zeros(B, max(N, cluster_indices_flat.shape[1]), C_dim, device=V.device)
        W_nodes = W_nodes.scatter(-2, cluster_indices_flat, W_expanded)
        W_nodes = W_nodes[:, :N] 

        nodes = torch.cat([V, W_nodes, positional_encoding], dim=-1)

        nodes, _ = self.gnn(nodes, E, edges)

        final_state = self.final_mlp(nodes)
        return final_state
