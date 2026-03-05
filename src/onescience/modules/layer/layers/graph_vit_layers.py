import torch
import torch.nn as nn
from torch_scatter import scatter_sum
from onescience.modules.mlp import StandardMLP as MLP

class GNN(nn.Module):
    """
    图神经网络层 (Graph Neural Network Layer)
    基于消息传递机制 (Message Passing) 更新节点和边特征。
    """
    def __init__(self, n_hidden=2, node_size=128, edge_size=128, output_size=None, layer_norm=False):
        super(GNN, self).__init__()
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

# ==========================================
# 核心层组件 (Core GraphViT Layers)
# ==========================================

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
            input_dim=4, 
            hidden_dims=[_hidden_dim], # n_hidden=1
            output_dim=128, 
            activation='relu',
            norm_layer=None
        )

        node_size = 128 + pos_length * 8
        
        self.encoder_gn = nn.ModuleList(
            [
                GNN(
                    node_size=node_size, edge_size=128, output_size=128, layer_norm=True
                )
                for _ in range(nb_gn)
            ]
        )

    def forward(self, mesh_pos, edges, states, node_type, pos_enc):
        V = torch.cat([states, node_type], dim=-1)

        senders = torch.gather(
            mesh_pos, -2, edges[..., 0].unsqueeze(-1).repeat(1, 1, 3) 
        )
        receivers = torch.gather(
            mesh_pos, -2, edges[..., 1].unsqueeze(-1).repeat(1, 1, 3)
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


class ClusterPoolingRNN(nn.Module):
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
            >>> pool = ClusterPoolingRNN(w_size=512, pos_length=7)
            >>> # W = pool(V, clusters, pos_enc, mask)
            >>> # W.shape -> torch.Size([1, K, 512])
    """
    def __init__(self, w_size, pos_length):
        super(ClusterPoolingRNN, self).__init__()
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


class ClusterToNodeDecoder(nn.Module):
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
            >>> decoder = ClusterToNodeDecoder(w_size=512, pos_length=7, state_size=2)
            >>> # out = decoder(W, V, clusters, pos_enc, edges, E)
            >>> # out.shape -> torch.Size([1, N, 2])
    """
    def __init__(self, w_size, pos_length, state_size):
        super(ClusterToNodeDecoder, self).__init__()
        pos_size = pos_length * 8
        node_size = w_size + 128 + pos_size
        
        self.gnn = GNN(node_size=node_size, output_size=128)
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


class TransformerBlockPreLN(nn.Module):
    """
        带有预归一化（Pre-LayerNorm）的 Transformer 注意力块。

        该模块用于在隐空间（簇级别）进行全局信息交互。
        它包含多头自注意力机制（MultiheadAttention）和一个前馈网络（MLP），并采用残差连接。
        输入特征在进入注意力层之前会与位置编码拼接。

        Args:
            w_size (int): 输入和输出的特征维度。
            pos_length (int): 位置编码长度，用于计算拼接后的嵌入维度。
            n_heads (int): 多头注意力的头数。

        形状:
            输入 W: (B, K, w_size)，簇特征序列。
            输入 attention_mask: (B * n_heads, K, K)，注意力掩码。
            输入 posenc: (B, K, P)，簇中心的位置编码。
            输出 W_new: (B, K, w_size)，更新后的簇特征。

        Example:
            >>> attn = TransformerBlockPreLN(w_size=512, pos_length=7, n_heads=4)
            >>> # W_new = attn(W, mask, pos_enc)
    """
    def __init__(self, w_size, pos_length, n_heads):    
        super(TransformerBlockPreLN, self).__init__()
        self.ln1 = nn.LayerNorm(w_size)

        embed_dim = w_size + 4 * pos_length 
        
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=n_heads, batch_first=True
        )
        self.linear = nn.Linear(embed_dim, w_size)
        self.ln2 = nn.LayerNorm(w_size)

        self.mlp = MLP(
            input_dim=w_size,
            hidden_dims=[w_size], 
            output_dim=w_size,
            activation='relu',
            norm_layer=None
        )

    def forward(self, W, attention_mask, posenc):
        # Pre-LayerNorm
        W1 = self.ln1(W)

        # 注入位置编码 (Concat 方式)
        W1_posenc = torch.cat([W1, posenc], dim=-1)

        # Self-Attention
        attn_out = self.attention(W1_posenc, W1_posenc, W1_posenc, attn_mask=attention_mask)[0]
        
        W3 = W + self.linear(attn_out)
        W4 = self.ln2(W3)
        W5 = self.mlp(W4)
        W6 = W3 + W5

        return W6


class FourierPosEncoder(nn.Module):
    """
        傅里叶特征位置编码器。

        该模块生成基于坐标的傅里叶位置编码。它不仅计算节点的绝对位置编码，还计算节点相对于其所属簇中心的相对位置编码。
        编码公式类似于 NeRF 中的位置编码：gamma(p) = [cos(2^k * pi * p), sin(2^k * pi * p)]。

        Args:
            pos_start (int): 频率指数的起始值（控制最低频率）。
            pos_length (int): 频率带的数量（控制频带宽度和输出维度）。

        形状:
            输入 mesh_pos: (B, N, D)，节点坐标。
            输入 clusters: (B, K, C_max)，簇分配索引。
            输入 cluster_mask: (B, K, C_max)，簇掩码。
            输出 nodes_embedding: (B, N, P_total)，包含绝对位置和相对位置的节点编码。
            输出 cluster_embedding: (B, K, P_embed)，簇中心的绝对位置编码。

        Example:
            >>> pos_enc = FourierPosEncoder(pos_start=-3, pos_length=8)
            >>> # node_emb, cluster_emb = pos_enc(pos, clusters, mask)
    """
    def __init__(self, pos_start, pos_length):
        super(FourierPosEncoder, self).__init__()
        self.pos_length = pos_length
        self.pos_start = pos_start

    def forward(self, mesh_pos, clusters, cluster_mask):
        """
        参数:
            mesh_pos: 节点坐标 [B, N, D]
            clusters: 簇分配索引 [B, K, C_max]
            cluster_mask: 簇掩码
        返回:
            nodes_embedding: 节点位置特征 (绝对+相对) [B, N, P_total]
            cluster_embedding: 簇中心位置特征 [B, K, P_embed]
        """
        B, N, _ = mesh_pos.shape
        _, K, C_max = clusters.shape

        # 收集簇内节点的坐标
        meshpos_gather_idx = clusters.reshape(B, -1, 1).repeat(1, 1, mesh_pos.shape[-1])
        meshpos_by_cluster = torch.gather(mesh_pos, -2, meshpos_gather_idx)
        meshpos_by_cluster = meshpos_by_cluster.reshape(*clusters.shape, -1)

        # 计算簇中心 
        clusters_centers = meshpos_by_cluster.sum(dim=-2)
        clusters_centers = clusters_centers / (
            cluster_mask.sum(-1, keepdim=True) + 1e-8
        )

        # 计算相对距离
        distances_to_cluster = clusters_centers.unsqueeze(-2) - meshpos_by_cluster
        
        # 编码相对位置
        pos_embeddings = self.embed(distances_to_cluster)
        S = pos_embeddings.shape[-1]
        
        # 将相对位置编码 Scatter 回节点顺序
        pos_embeddings_flat = pos_embeddings.reshape(B, -1, S)
        cluster_indices_flat = clusters.reshape(B, -1, 1).repeat(1, 1, S)
        
        relative_positions = torch.zeros(B, max(N, cluster_indices_flat.shape[1]), S, device=mesh_pos.device)
        relative_positions = relative_positions.scatter(
            -2,
            cluster_indices_flat,
            pos_embeddings_flat,
        )
        relative_positions = relative_positions[:, :N]

        # 编码绝对位置并拼接
        nodes_embedding = torch.cat([self.embed(mesh_pos), relative_positions], dim=-1)

        return nodes_embedding, self.embed(clusters_centers)

    def embed(self, pos):
        """
        傅里叶特征映射函数
        Formula: [cos(2^k * pi * p), sin(2^k * pi * p)]
        """
        original_shape = pos.shape
        # 展平以便处理任意维度的输入
        pos = pos.reshape(-1, original_shape[-1])
        
        index = torch.arange(
            self.pos_start, self.pos_start + self.pos_length, device=pos.device
        )
        index = index.float()
        freq = 2**index * torch.pi # [Length]
        
        # 计算频率项: pos [M, D] * freq [L] -> [M, D, L]
        args = freq.view(1, 1, -1) * pos.unsqueeze(-1)
        
        # 计算 sin 和 cos
        cos_feat = torch.cos(args)
        sin_feat = torch.sin(args)
        
        # 拼接: [M, D, 2*L]
        embedding = torch.cat([cos_feat, sin_feat], dim=-1)
        
        # 恢复原始形状，最后维度变为 D * 2 * Length
        embedding = embedding.view(*original_shape[:-1], -1)
        return embedding