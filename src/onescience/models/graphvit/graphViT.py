import torch
import torch.nn as nn

# 从统一的模块工厂导入组件
from onescience.modules import (
    OneEncoder,
    OneDecoder,
    OnePooling,
    OneTransformer,
    OneEmbedding
)

# 节点类型常量
NODE_NORMAL = 0
NODE_INPUT = 4
NODE_OUTPUT = 5
NODE_WALL = 6
NODE_DISABLE = 2

class GraphViT(nn.Module):
    """

    该模型利用图神经网络提取局部特征，通过聚类池化在潜在空间进行全局 Transformer 交互，
    最后解码回物理空间进行下一时刻的状态预测。

    Args:
        state_size (int): 物理状态的维度（如速度 u, v, w）。
        w_size (int): 潜在空间（Cluster Level）的特征维度。默认值: 512。
        n_attention (int): Transformer 注意力层的数量。默认值: 4。
        nb_gn (int): Encoder 中 GNN 的层数。默认值: 4。
        n_heads (int): Transformer 注意力头数。默认值: 4。
    """
    def __init__(self, state_size, w_size=512, n_attention=4, nb_gn=4, n_heads=4):
        super(GraphViT, self).__init__()
        
        # 傅里叶位置编码参数
        pos_start = -3
        pos_length = 8

        # 1. 编码器: Mesh Space -> Latent Graph
        self.encoder = OneEncoder(
            style="GraphViTEncoder",
            nb_gn=nb_gn, 
            state_size=state_size, 
            pos_length=pos_length
        )

        # 2. 池化层: Node Features -> Cluster Features
        self.graph_pooling = OnePooling(
            style="RNNClusterPooling",
            w_size=w_size, 
            pos_length=pos_length
        )

        # 3. 解码器: Cluster Features -> Node State Update
        self.graph_retrieve = OneDecoder(
            style="GraphViTDecoder",
            w_size=w_size, 
            pos_length=pos_length, 
            state_size=state_size
        )

        # 4. 潜在空间 Transformer 交互层
        self.attention = nn.ModuleList([
            OneTransformer(
                style="PreLNTransformerBlock", 
                w_size=w_size, 
                pos_length=pos_length, 
                n_heads=n_heads
            )
            for _ in range(n_attention)
        ])
        
        self.ln = nn.LayerNorm(w_size)
        self.noise_std = 0.0
        
        # 5. 坐标嵌入层
        self.positional_encoder = OneEmbedding(
            style="FourierPosEmbedding",
            pos_start=pos_start, 
            pos_length=pos_length
        )

    def forward(
        self,
        mesh_pos,
        edges,
        state,
        node_type,
        clusters,
        clusters_mask,
        apply_noise=False,
    ):
        """
        前向传播 (自回归滚动预测)。
        
        参数:
            mesh_pos: (B, T, N, 3) 节点坐标
            edges: (B, T, M, 2) 边索引
            state: (B, T, N, state_size) 节点物理状态
            node_type: (B, T, N, 9) 节点类型 (One-hot)
            clusters: (B, T, K, C_max) 簇索引
            clusters_mask: (B, T, K, C_max) 簇掩码
            apply_noise: 是否在初始状态施加噪声 (Training 时通常为 True)
        """
        if apply_noise:
            mask = torch.logical_or(
                node_type[:, 0, :, NODE_NORMAL] == 1,
                node_type[:, 0, :, NODE_OUTPUT] == 1,
            )
            noise = torch.randn_like(state[:, 0]).to(state.device) * self.noise_std
            state[:, 0][mask] = state[:, 0][mask] + noise[mask]

        state_hat = [state[:, 0]] 
        output_hat = []
        target = []

        # 从 t=1 开始预测，基于 t-1 的状态
        for t in range(1, state.shape[1]):
            # 1. 生成位置编码
            mesh_posenc, cluster_posenc = self.positional_encoder(
                mesh_pos[:, t - 1], clusters[:, t - 1], clusters_mask[:, t - 1]
            )

            # 2. 图节点和边编码
            V, E = self.encoder(
                mesh_pos[:, t - 1],
                edges[:, t - 1],
                state_hat[-1],
                node_type[:, t - 1],
                mesh_posenc,
            )

            # 3. 聚类池化
            W = self.graph_pooling(
                V, clusters[:, t - 1], mesh_posenc, clusters_mask[:, t - 1]
            )

            # 4. 构建 Attention Mask
            attention_mask = clusters_mask[:, t - 1].sum(-1, keepdim=True) == 0
            attention_mask = (
                attention_mask.unsqueeze(1)
                .repeat(1, len(self.attention), 1, W.shape[1])
                .view(-1, W.shape[1], W.shape[1])
            )
            attention_mask[:, torch.eye(W.shape[1], dtype=torch.bool)] = False
            attention_mask = attention_mask.transpose(-1, -2)

            # 5. 潜在空间 Transformer 交互
            for a in self.attention:
                W = a(W, attention_mask, cluster_posenc)
            W = self.ln(W)

            # 6. 解码预测更新量
            next_output = self.graph_retrieve(
                W, V, clusters[:, t - 1], mesh_posenc, edges[:, t - 1], E
            )

            # 7. 更新下一时刻状态
            next_state = state_hat[-1] + next_output
            target.append(state[:, t] - state_hat[-1])

            # 8. 强制边界条件 (Mask 覆盖)
            mask = torch.logical_or(
                node_type[:, t, :, NODE_INPUT] == 1, node_type[:, t, :, NODE_WALL] == 1
            )
            mask = torch.logical_or(mask, node_type[:, t, :, NODE_DISABLE] == 1)
            next_state[mask, :] = state[:, t][mask, :]

            state_hat.append(next_state)
            output_hat.append(next_output)

        # 堆叠时间步
        velocity_hat = torch.stack(state_hat, dim=1)
        output_hat = torch.stack(output_hat, dim=1)
        target = torch.stack(target, dim=1)

        # velocity_hat: [B, T, N, S], output_hat: [B, T-1, N, S]
        return velocity_hat, output_hat, target