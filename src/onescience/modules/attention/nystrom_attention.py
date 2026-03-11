from math import ceil
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce

# helper functions
def exists(val):
    return val is not None

def moore_penrose_iter_pinv(x, iters = 6):
    device = x.device

    abs_x = torch.abs(x)
    col = abs_x.sum(dim = -1)
    row = abs_x.sum(dim = -2)
    z = rearrange(x, '... i j -> ... j i') / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-1], device = device)
    I = rearrange(I, 'i j -> () i j')

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

    return z

# main attention class
class NystromAttention(nn.Module):
    """
    Nystrom 注意力机制 (Nyström Attention)。

    该模块通过 Nyström 方法对标准自注意力矩阵进行低秩近似，从而将 Transformer 的
    时间与空间复杂度从 $O(N^2)$ 降低到 $O(N)$。它通过提取少量的 Landmark (地标节点) 
    来重构全局的注意力矩阵，并使用 Moore-Penrose 伪逆的迭代逼近算法来保证数值稳定性。
    非常适用于处理超长序列或高分辨率网格的物理场数据。

    Args:
        dim (int): 输入和输出的特征维度。
        dim_head (int, optional): 每个注意力头的维度。默认值: 64。
        heads (int, optional): 注意力头的数量。默认值: 8。
        num_landmarks (int, optional): 用于近似的地标节点数量。默认值: 256。
        pinv_iterations (int, optional): 伪逆迭代逼近的次数。默认值: 6。
        residual (bool, optional): 是否在 Value 上添加深度卷积残差。默认值: True。
        residual_conv_kernel (int, optional): 残差卷积的核大小。默认值: 33。
        eps (float, optional): 防止除零的极小值。默认值: 1e-8。
        dropout (float, optional): Dropout 概率。默认值: 0.0。

    形状:
        输入 x: (B, N, C)，其中 N 为序列长度，C 为特征维度 (dim)。
        输入 mask: (B, N)，布尔类型的掩码。
        输出 out: (B, N, C)，形状与输入 x 保持一致。

    Example:
        >>> attn = NystromAttention(dim=128, heads=4, num_landmarks=64)
        >>> x = torch.randn(2, 1024, 128)
        >>> out = attn(x)
        >>> out.shape
        torch.Size([2, 1024, 128])
    """
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        num_landmarks = 256,
        pinv_iterations = 6,
        residual = True,
        residual_conv_kernel = 33,
        eps = 1e-8,
        dropout = 0.
    ):
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head

        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding = (padding, 0), groups = heads, bias = False)

    def forward(self, x, mask = None, return_attn = False, return_attn_matrices = False):
        b, n, _, h, m, iters, eps = *x.shape, self.heads, self.num_landmarks, self.pinv_iterations, self.eps

        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            x = F.pad(x, (0, 0, padding, 0), value = 0)

            if exists(mask):
                mask = F.pad(mask, (padding, 0), value = False)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        if exists(mask):
            mask = rearrange(mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        q = q * self.scale

        l = ceil(n / m)
        landmark_einops_eq = '... (n l) d -> ... n d'
        q_landmarks = reduce(q, landmark_einops_eq, 'sum', l = l)
        k_landmarks = reduce(k, landmark_einops_eq, 'sum', l = l)

        divisor = l
        if exists(mask):
            mask_landmarks_sum = reduce(mask, '... (n l) -> ... n', 'sum', l = l)
            divisor = mask_landmarks_sum[..., None] + eps
            mask_landmarks = mask_landmarks_sum > 0

        q_landmarks = q_landmarks / divisor
        k_landmarks = k_landmarks / divisor

        einops_eq = '... i d, ... j d -> ... i j'
        sim1 = einsum(einops_eq, q, k_landmarks)
        sim2 = einsum(einops_eq, q_landmarks, k_landmarks)
        sim3 = einsum(einops_eq, q_landmarks, k)

        if exists(mask):
            mask_value = -torch.finfo(q.dtype).max
            sim1.masked_fill_(~(mask[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim2.masked_fill_(~(mask_landmarks[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim3.masked_fill_(~(mask_landmarks[..., None] * mask[..., None, :]), mask_value)

        attn1, attn2, attn3 = map(lambda t: t.softmax(dim = -1), (sim1, sim2, sim3))
        attn2_inv = moore_penrose_iter_pinv(attn2, iters)

        out = (attn1 @ attn2_inv) @ (attn3 @ v)

        if self.residual:
            out = out + self.res_conv(v)

        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)
        out = out[:, -n:]

        if return_attn_matrices:
            return out, (attn1, attn2_inv, attn3)
        elif return_attn:
            attn = attn1 @ attn2_inv @ attn3
            return out, attn

        return out