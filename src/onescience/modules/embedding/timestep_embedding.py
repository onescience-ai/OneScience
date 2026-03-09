import math
import torch
import torch.nn as nn
from einops import rearrange
import numpy as np

def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
        创建正弦时间步长嵌入 (Sinusoidal Timestep Embeddings)。

        该函数类似于 Transformer 中的位置编码，用于将标量时间步（或噪声水平）转换为高维向量表示。它使用不同频率的正弦和余弦函数对输入进行编码。
        计算公式如下：
        PE(t, 2i) = sin(t / 10000^(2i/dim))
        PE(t, 2i+1) = cos(t / 10000^(2i/dim))

        Args:
            timesteps (Tensor): 一维张量，包含 N 个时间步索引（可以是分数）。形状为 (N,)。
            dim (int): 输出嵌入的维度。
            max_period (int, optional): 控制嵌入的最小频率（最大周期）。默认值: 10000。
            repeat_only (bool, optional): 代码中保留参数，但在当前实现中未使用。默认值: False。

        形状:
            输入: (N,)
            输出: (N, dim)

        Example:
            >>> t = torch.arange(0, 10)
            >>> emb = timestep_embedding(t, dim=128)
            >>> emb.shape
            torch.Size([10, 128])
    """

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:,:,:1])], dim=-1)
    return embedding

