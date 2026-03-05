import torch
import torch.nn as nn
from onescience.modules.mlp import StandardMLP as MLP

class PreLNTransformerBlock(nn.Module):
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
            >>> attn = PreLNTransformerBlock(w_size=512, pos_length=7, n_heads=4)
            >>> # W_new = attn(W, mask, pos_enc)
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