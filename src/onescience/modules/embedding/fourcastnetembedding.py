import torch
import torch.nn as nn

class FourCastNetEmbedding(nn.Module):
    """
        FourCastNet 的 2D Patch Embedding 模块。

        使用 2D 卷积将气象场图像划分为不重叠的 Patch 并投影到嵌入空间，
        是 FourCastNet 编码器的入口层。与 FuxiEmbedding 的 3D 卷积不同，
        该模块仅处理单帧二维气象场，输出展平为序列形式供后续 Transformer 使用。

        Args:
            img_size (tuple[int, int], optional): 输入气象场的空间分辨率 (lat, lon)，
                默认为 (720, 1440)。
            patch_size (tuple[int, int], optional): Patch 大小 (Plat, Plon)，
                默认为 (8, 8)。
            in_chans (int, optional): 输入气象变量的通道数，默认为 19。
            embed_dim (int, optional): Patch 嵌入维度，默认为 768。

        形状:
            - 输入 x: (B, C, lat, lon)，其中 C = in_chans
            - 输出:   (B, num_patches, embed_dim)
                其中 num_patches = (lat // Plat) * (lon // Plon)

        Examples:
            >>> # 典型 FourCastNet 配置
            >>> # 分辨率 720×1440，Patch 大小 8×8
            >>> # num_patches = (720//8) * (1440//8) = 90 * 180 = 16200
            >>> embedding = FourCastNetEmbedding(
            ...     img_size=(720, 1440),
            ...     patch_size=(8, 8),
            ...     in_chans=19,
            ...     embed_dim=768,
            ... )
            >>> x = torch.randn(2, 19, 720, 1440)  # (B, C, lat, lon)
            >>> out = embedding(x)
            >>> out.shape
            torch.Size([2, 16200, 768])
    """
    def __init__(self, 
                 img_size=(720, 1440), 
                 patch_size=(8, 8), 
                 in_chans=19, 
                 embed_dim=768):
        super().__init__()
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

