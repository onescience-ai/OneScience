import torch
import torch.nn as nn

class FourierPosEmbedding(nn.Module):
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
            >>> pos_enc = FourierPosEmbedding(pos_start=-3, pos_length=8)
            >>> # node_emb, cluster_emb = pos_enc(pos, clusters, mask)
    """
    def __init__(self, pos_start, pos_length):
        super(FourierPosEmbedding, self).__init__()
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