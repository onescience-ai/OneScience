import torch
import torch.nn as nn
from timm.layers import trunc_normal_
from typing import Optional, Tuple

class WindowAttention(nn.Module):
    r"""
    基于窗口的多头自注意力机制 (Window-based Multi-Head Self-Attention, W-MSA)，带相对位置偏置。

    这是 Swin Transformer 的核心注意力模块。与标准的全局自注意力不同，它仅在局部的窗口内计算注意力。
    这种设计使得计算复杂度与图像尺寸呈线性关系。
    该模块同时支持移动窗口（Shifted Window）和非移动窗口配置（通过外部传入的 `mask` 参数控制）。
    此外，它还包含了一个可学习的相对位置偏置（Relative Position Bias），这已被证明能显著提升视觉任务的性能。

    Args:
        dim (int): 输入通道数（特征维度）。
        window_size (tuple[int]): 窗口的高度和宽度 (Wh, Ww)。
        num_heads (int): 注意力头的数量。
        qkv_bias (bool, optional): 如果为 True，则为 Query, Key, Value 的投影层添加可学习的偏置。默认值: True。
        qk_scale (float | None, optional): 如果设置，将覆盖默认的缩放因子 (head_dim ** -0.5)。
        attn_drop (float, optional): 注意力权重的 Dropout 比率。默认值: 0.0。
        proj_drop (float, optional): 输出层的 Dropout 比率。默认值: 0.0。

    形状:
        - 输入: :math:`(B \times N_{windows}, N, C)`，其中 :math:`N = Wh \times Ww` 是窗口内的 token 数量。
        - 掩码 (可选): :math:`(N_{windows}, N, N)` 或 :math:`(1, N, N)`。
        - 输出: :math:`(B \times N_{windows}, N, C)`。

    Example:
        >>> # 假设输入特征图大小 (H, W) = (14, 14)，窗口大小 = 7
        >>> # Batch size = 1, Channels = 96, Heads = 3
        >>> # 窗口数量 = (14/7) * (14/7) = 4
        >>> attn = WindowAttention(dim = 96, window_size = (7, 7), num_heads = 3)
        >>>
        >>> # 输入形状: (Batch * Num_Windows, Window_Area, Channels)
        >>> # Window_Area = 7 * 7 = 49
        >>> x = torch.randn(1 * 4, 49, 96)
        >>> out = attn(x)
        >>> print(out.shape)
        torch.Size([4, 49, 96])
    """

    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # 定义相对位置偏置参数表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH

        # 获取窗口内每个 token 的成对相对位置索引
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        
        # 明确指定 indexing='ij' 以避免 PyTorch 未来版本的警告，并确保坐标生成的确定性
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: 输入特征，形状为 (num_windows*B, N, C)
            mask: (0/-inf) 掩码，形状为 (num_windows, Wh*Ww, Wh*Ww) 或 None
        """
        B_, N, C = x.shape
        
        # 维度检查
        if C != self.dim:
            raise ValueError(f"Input feature dimension {C} does not match initialized dim {self.dim}")
        
        # 序列长度检查
        expected_N = self.window_size[0] * self.window_size[1]
        if N != expected_N:
             raise ValueError(f"Input sequence length {N} does not match window size {self.window_size} (expected {expected_N})")

        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            # 增强点 4: 掩码与批次大小的兼容性检查
            # 确保总批次大小 (B_) 能够被窗口数量 (nW) 整除，否则 view 操作会报错且难以调试
            if B_ % nW != 0:
                 raise ValueError(f"Batch size {B_} is not divisible by number of windows in mask {nW}")
                 
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops