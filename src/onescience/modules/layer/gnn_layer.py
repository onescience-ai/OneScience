import torch
import torch.nn as nn
from torch_scatter import scatter_sum
from onescience.modules.mlp import StandardMLP as MLP

class GNNLayer(nn.Module):
    """
    图神经网络层 (Graph Neural Network Layer)
    基于消息传递机制 (Message Passing) 更新节点和边特征。
    """
    def __init__(self, n_hidden=2, node_size=128, edge_size=128, output_size=None, layer_norm=False):
        super(GNNLayer, self).__init__()
        output_size = output_size or node_size
        _hidden_dim = 128 
        edge_mlp_hidden_dims = [_hidden_dim] * n_hidden
        node_mlp_hidden_dims = [_hidden_dim] * n_hidden

        # 边更新网络
        self.f_edge = MLP(
            input_dim=edge_size + node_size * 2, 
            hidden_dims=edge_mlp_hidden_dims,
            output_dim=edge_size,
            activation='relu',
            norm_layer=None, 
            use_bias=True
        )
        self.edge_norm = nn.LayerNorm(edge_size) if layer_norm else nn.Identity()

        # 节点更新网络
        self.f_node = MLP(
            input_dim=edge_size + node_size, 
            hidden_dims=node_mlp_hidden_dims,
            output_dim=output_size,
            activation='relu',
            norm_layer=None,
            use_bias=True
        )
        self.node_norm = nn.LayerNorm(output_size) if layer_norm else nn.Identity()

    def forward(self, V, E, edges):
        # 收集特征
        senders = torch.gather(V, -2, edges[..., 0].unsqueeze(-1).repeat(1, 1, V.shape[-1]))
        receivers = torch.gather(V, -2, edges[..., 1].unsqueeze(-1).repeat(1, 1, V.shape[-1]))

        # 更新边
        edge_inpt = torch.cat([senders, receivers, E], dim=-1)
        edge_embeddings = self.edge_norm(self.f_edge(edge_inpt))

        # 聚合
        col = edges[..., 0].unsqueeze(-1).repeat(1, 1, edge_embeddings.shape[-1])
        edge_sum = scatter_sum(edge_embeddings, col, dim=-2, dim_size=V.shape[-2])

        # 更新节点
        node_inpt = torch.cat([V, edge_sum], dim=-1)
        node_embeddings = self.node_norm(self.f_node(node_inpt))

        return node_embeddings, edge_embeddings