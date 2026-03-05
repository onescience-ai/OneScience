import torch
import torch.nn as nn
from onescience.modules.mlp import StandardMLP as MLP
from onescience.modules.layer.gnn_layer import GNNLayer

class GraphViTEncoder(nn.Module):
    """
        基于图神经网络（GNN）的编码器模块。

        该模块负责将输入的物理状态（位置、速度、节点类型）编码为隐空间的节点特征（V）和边特征（E）。
        它首先通过 MLP 对节点和边进行独立编码，然后通过多层 GNN 进行消息传递和特征更新。

        Args:
            nb_gn (int, optional): GNN 层的堆叠数量（消息传递次数）。默认值: 4。
            state_size (int, optional): 输入物理状态的维度（例如速度的维度）。默认值: 3。
            pos_length (int, optional): 位置编码的频带数量，用于计算输入节点特征的维度。默认值: 7。

        形状:
            输入 mesh_pos: (B, N, D)，节点坐标。
            输入 edges: (B, M, 2)，边索引列表。
            输入 states: (B, N, S_in)，节点物理状态。
            输入 node_type: (B, N, T_type)，节点类型 One-hot 编码。
            输入 pos_enc: (B, N, P)，节点位置编码。
            输出 V: (B, N, 128)，编码后的节点特征。
            输出 E: (B, M, 128)，编码后的边特征。

        Example:
            >>> encoder = GraphViTEncoder(nb_gn=2, state_size=2)
            >>> # 假设 B=1, N=100, M=200
            >>> # V, E = encoder(mesh_pos, edges, states, node_type, pos_enc)
            >>> # V.shape -> torch.Size([1, 100, 128])
    """
    def __init__(self, nb_gn=4, state_size=3, pos_length=7):
        super(GraphViTEncoder, self).__init__()
        _hidden_dim = 128
        
        self.encoder_node = MLP(
            input_dim=9 + state_size, 
            hidden_dims=[_hidden_dim], # n_hidden=1
            output_dim=128, 
            activation='relu',
            norm_layer=None
        )
        
        self.encoder_edge = MLP(
            input_dim=3, 
            hidden_dims=[_hidden_dim], # n_hidden=1
            output_dim=128, 
            activation='relu',
            norm_layer=None
        )

        node_size = 128 + pos_length * 8
        
        self.encoder_gn = nn.ModuleList(
            [
                GNNLayer(
                    node_size=node_size, edge_size=128, output_size=128, layer_norm=True
                )
                for _ in range(nb_gn)
            ]
        )

    def forward(self, mesh_pos, edges, states, node_type, pos_enc):
        V = torch.cat([states, node_type], dim=-1)

        senders = torch.gather(
            mesh_pos, -2, edges[..., 0].unsqueeze(-1).repeat(1, 1, 2) 
        )
        receivers = torch.gather(
            mesh_pos, -2, edges[..., 1].unsqueeze(-1).repeat(1, 1, 2)
        )
        distance = senders - receivers
        norm = torch.sqrt((distance**2).sum(-1, keepdims=True))
        E = torch.cat([distance, norm], dim=-1)

        V = self.encoder_node(V)
        E = self.encoder_edge(E) 

        for i in range(len(self.encoder_gn)):
            inpt = torch.cat([V, pos_enc], dim=-1)
            v, e = self.encoder_gn[i](inpt, E, edges)
            V = V + v 
            E = E + e 

        return V, E