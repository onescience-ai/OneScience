import torch
import torch.nn as nn
from onescience.modules.mlp.onemlp import OneMlp

class PreLNTransformerBlock(nn.Module):
    """
    带有预归一化 (Pre-LayerNorm) 的 Transformer 注意力块。

    该模块通常用于在潜空间 (Latent Space) 或簇级别 (Cluster-level) 进行全局信息的交互与路由。
    它采用标准的 Pre-Norm 结构，包含多头自注意力机制 (MultiheadAttention) 和一个前馈神经网络 (MLP)，
    并在两个子层周围应用了残差连接。
    特别地，该模块在计算注意力之前，会将输入的特征与显式的位置编码在通道维度上进行拼接 (Concat)，
    以增强模型对空间结构与物理拓扑的感知能力。

    Args:
        w_size (int): 输入和输出的特征维度 (即簇特征的通道数)。
        pos_length (int): 基础位置编码的长度。注意：模块内部假定传入的实际位置编码特征维度为 4 * pos_length。
        n_heads (int): 多头注意力机制的头数。注意：拼接后的总维度 (w_size + 4 * pos_length) 必须能被 n_heads 整除。

    形状:
        输入 W: (B, K, w_size)，其中 B 为 Batch Size，K 为簇 (或序列节点) 的数量。
        输入 attention_mask: (B * n_heads, K, K) 或 (K, K)，用于多头注意力的掩码矩阵。
        输入 posenc: (B, K, 4 * pos_length)，待拼接的位置编码特征。
        输出 W_new: (B, K, w_size)，更新后的特征张量。

    Example:
        >>> block = PreLNTransformerBlock(w_size=128, pos_length=16, n_heads=8)
        >>> W = torch.randn(2, 64, 128)          # B=2, K=64
        >>> mask = torch.zeros(2 * 8, 64, 64)    # attention mask
        >>> pos_enc = torch.randn(2, 64, 64)     # 4 * pos_length = 64
        >>> out = block(W, mask, pos_enc)
        >>> print(out.shape)
        torch.Size([2, 64, 128])
    """
    def __init__(self, w_size, pos_length, n_heads):    
        super(PreLNTransformerBlock, self).__init__()
        self.ln1 = nn.LayerNorm(w_size)

        embed_dim = w_size + 4 * pos_length 
        
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=n_heads, batch_first=True
        )
        self.linear = nn.Linear(embed_dim, w_size)
        self.ln2 = nn.LayerNorm(w_size)

        self.mlp = OneMlp(
            style="StandardMLP",
            input_dim=w_size,
            hidden_dims=[w_size], 
            output_dim=w_size,
            activation='relu',
        )

    def forward(self, W, attention_mask, posenc):
        W1 = self.ln1(W)
        W1_posenc = torch.cat([W1, posenc], dim=-1)

        attn_out = self.attention(W1_posenc, W1_posenc, W1_posenc, attn_mask=attention_mask)[0]
        
        W3 = W + self.linear(attn_out)
        W4 = self.ln2(W3)
        W5 = self.mlp(W4)
        W6 = W3 + W5

        return W6