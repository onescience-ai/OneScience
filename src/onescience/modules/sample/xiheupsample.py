import torch
import torch.nn.functional as F
from torch import nn

class XiheUpSample(nn.Module):
    """
        xihe 风格的 2D 空间上采样模块。
        
        XiheDownSample2D 的逆操作。先通过线性层将通道数扩展（C → out_dim * 4），
        再将每个 token 拆分为 2×2 的子像素块（类似 PixelShuffle），空间分辨率扩大为
        原来的 2 倍。当上采样后的分辨率超出目标分辨率时，自动通过裁剪对齐
        output_resolution。
        
        Args:
            in_dim (int): 输入 token 的通道数。
            out_dim (int): 输出 token 的通道数，线性层先扩展至 out_dim * 4，
                拆分后每个子像素通道数恢复为 out_dim。
            input_resolution (tuple[int, int]): 输入特征图的空间分辨率 (lat, lon)。
            output_resolution (tuple[int, int]): 目标输出分辨率 (out_lat, out_lon)，
                应满足 out_lat ≤ in_lat * 2 且 out_lon ≤ in_lon * 2，超出部分通过
                中心裁剪去除。
        
        形状:
            - 输入 x: (B, lat * lon, C)，其中 C = in_dim
            - 输出:   (B, out_lat * out_lon, out_dim)
        
        Examples:
            >>> # 气象场分辨率 91×180 → 181×360 上采样（对应 PanguDownSample2D 的逆操作）
            >>> # in_lat * 2 = 91 * 2 = 182，裁剪掉多余的1行: pad_h = 182 - 181 = 1
            >>> # in_lon * 2 = 180 * 2 = 360，无需裁剪: pad_w = 0
            >>> # 输入 token 数:  91 * 180 = 16380
            >>> # 输出 token 数: 181 * 360 = 65160
            >>> upsample = PanguUpSample2D(
            ...     in_dim=384,
            ...     out_dim=192,
            ...     input_resolution=(91, 180),
            ...     output_resolution=(181, 360),
            ... )
            >>> x = torch.randn(2, 16380, 384)  # (B, lat*lon, C)
            >>> out = upsample(x)
            >>> out.shape
            torch.Size([2, 65160, 192])
            
            >>> # 整除情况下无需裁剪（如 64×128 → 128×256）
            >>> upsample2 = PanguUpSample2D(
            ...     in_dim=384,
            ...     out_dim=192,
            ...     input_resolution=(64, 128),
            ...     output_resolution=(128, 256),
            ... )
            >>> x2 = torch.randn(2, 8192, 384)  # (B, 64*128, C)
            >>> out2 = upsample2(x2)
            >>> out2.shape
            torch.Size([2, 32768, 192])
    """
    def __init__(self, in_dim, out_dim, input_resolution, output_resolution):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim * 4, bias=False)
        self.linear2 = nn.Linear(out_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): (B, N, C)
        """
        B, N, C = x.shape
        in_lat, in_lon = self.input_resolution
        out_lat, out_lon = self.output_resolution

        x = self.linear1(x)
        x = x.reshape(B, in_lat, in_lon, 2, 2, C // 2).permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, in_lat * 2, in_lon * 2, -1)

        # ✅ 用插值精确拉伸到目标分辨率
        x = F.interpolate(
            x.permute(0, 3, 1, 2),
            size=(out_lat, out_lon),
            mode="bilinear",
            align_corners=False
        )
        x = x.permute(0, 2, 3, 1).reshape(B, out_lat * out_lon, -1)

        x = self.norm(x)
        x = self.linear2(x)
        return x