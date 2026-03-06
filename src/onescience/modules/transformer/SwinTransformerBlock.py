import torch
import torch.nn as nn
from timm.layers import DropPath, to_2tuple
from onescience.modules import OneMlp, OneAttention

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer 块 (Swin Transformer Block)。

    该模块实现了带有移动窗口机制的多头自注意力层。根据 `shift_size` 的设置，它可以作为：
    1. 标准的基于局部窗口的自注意力层 (W-MSA, `shift_size=0`)。
    2. 移动窗口自注意力层 (SW-MSA, `shift_size > 0`)。
    移动窗口机制能够限制注意力计算的复杂度为线性 ($O(N)$)，同时引入跨窗口的特征连接，非常适合处理二维网格或图像数据。

    Args:
        dim (int): 输入特征的通道数 (特征维度)。
        input_resolution (tuple[int]): 输入特征图的空间分辨率 (H, W)。
        num_heads (int): 注意力机制的头数。
        window_size (int, optional): 局部注意力窗口的大小。默认值: 7。
        shift_size (int, optional): 窗口移动的大小。对于 SW-MSA，通常设置为 window_size // 2。默认值: 0。
        mlp_ratio (float, optional): MLP 中隐藏层维度相对于输入维度的扩展比例。默认值: 4.0。
        qkv_bias (bool, optional): 是否为 Query, Key, Value 矩阵添加可学习的偏置。默认值: True。
        qk_scale (float | None, optional): 如果设置，将覆盖默认的 QK 缩放因子 (head_dim ** -0.5)。默认值: None。
        drop (float, optional): 特征投影后的 Dropout 概率。默认值: 0.0。
        attn_drop (float, optional): 注意力矩阵的 Dropout 概率。默认值: 0.0。
        drop_path (float, optional): Stochastic depth (随机深度) 丢弃概率，用于残差分支。默认值: 0.0。
        act_layer (nn.Module, optional): 激活函数层。默认值: nn.GELU。
        norm_layer (nn.Module, optional): 归一化层。默认值: nn.LayerNorm。
        fused_window_process (bool, optional): 是否使用融合的 CUDA 算子来加速窗口划分与循环移位操作。默认值: False。

    形状:
        输入 x: (B, L, C)，其中 L 是空间分辨率的展平长度 (L = H * W)，C 是特征维度 (dim)。
        输出: (B, L, C)，输出形状与输入保持一致。

    Example:
        >>> # 定义一个处理 14x14 网格，通道数为 64，窗口大小为 7 的 W-MSA 块
        >>> block = SwinTransformerBlock(
        ...     dim=64, 
        ...     input_resolution=(14, 14), 
        ...     num_heads=4, 
        ...     window_size=7, 
        ...     shift_size=0  
        ... )
        >>> # 输入形状为 (Batch_size, L, C)，这里 L = 14*14 = 196
        >>> x = torch.randn(2, 196, 64)
        >>> out = block(x)
        >>> print(out.shape)
        torch.Size([2, 196, 64])
    """
    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        fused_window_process=False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert (
            0 <= self.shift_size < self.window_size
        ), "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        
        self.attn = OneAttention(
            style="WindowAttention",
            dim=dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        self.mlp = OneMlp(
            style="StandardMLP",
            input_dim=dim,
            output_dim=dim,
            hidden_dims=[mlp_hidden_dim],
            activation="gelu", 
            use_bias=True
        )

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)
            ).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = torch.roll(
                    x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
                )
                x_windows = window_partition(shifted_x, self.window_size) 
        else:
            shifted_x = x
            x_windows = window_partition(shifted_x, self.window_size) 

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = window_reverse(attn_windows, self.window_size, H, W)
                x = torch.roll(
                    shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
                )
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)
            x = shifted_x
            
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x