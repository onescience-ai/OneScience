import torch
from torch import nn

class PanguEmbedding2D(nn.Module):
    """
        将二维图像分割为不重叠的 patch 并嵌入到向量空间。

        Args:
            img_size (tuple[int, int]): 输入图像尺寸 (H, W)
            patch_size (tuple[int, int]): 每个 patch 的大小 (patch_h, patch_w)
            in_chans (int): 输入图像通道数
            embed_dim (int): 每个 patch 嵌入后的向量维度
            norm_layer (nn.Module, optional): 归一化层，默认为 None。常用: nn.LayerNorm

        形状:
            输入: (B, C, H, W)
            输出: (B, embed_dim, H', W')，其中 H' = ⌈H / patch_h⌉, W' = ⌈W / patch_w⌉

        Example:
            >>> patch_embed = PatchEmbed2D(
            ...     img_size=(128, 256),
            ...     patch_size=(4, 4),
            ...     in_chans=3,
            ...     embed_dim=96
            ... )
            >>> x = torch.randn(8, 3, 128, 256)
            >>> out = patch_embed(x)
            >>> out.shape
            torch.Size([8, 96, 32, 64])
    """
        
    def __init__(self, img_size=(721, 1440),
                    patch_size=(4, 4),
                    embed_dim=192,
                    in_chans = 4+3,
                    norm_layer=None,
                    ):
        
        super().__init__()
        print(img_size)
        height, width = img_size
        h_patch_size, w_path_size = patch_size
        stride = patch_size
        padding_left = padding_right = padding_top = padding_bottom = 0
        h_remainder = height % h_patch_size
        w_remainder = width % w_path_size

        if h_remainder:
            h_pad = h_patch_size - h_remainder
            padding_top = h_pad // 2
            padding_bottom = int(h_pad - padding_top)

        if w_remainder:
            w_pad = w_path_size - w_remainder
            padding_left = w_pad // 2
            padding_right = int(w_pad - padding_left)

        self.pad = nn.ZeroPad2d(
            (padding_left, padding_right, padding_top, padding_bottom)
        )
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        x = self.pad(x)
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x