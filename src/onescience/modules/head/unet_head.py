import torch.nn as nn

class UNetHead1D(nn.Module):
    """
    一维 U-Net 预测头 (UNet Head 1D)。

    将解码器输出的深层特征映射到最终的物理量或分类通道。
    默认使用 1x1 卷积。如果修改 kernel_size > 1，会自动加入 padding 保证输出序列长度一致。

    Args:
        in_channels (int): 输入通道数 (通常等于 base_channels)。
        out_channels (int): 输出通道数 (即预测的物理量维度)。
        kernel_size (int, optional): 预测头卷积核大小。默认值: 1。

    形状:
        输入 x: (B, C_in, L)
        输出: (B, C_out, L)

    Example:
        >>> head = UNetHead1D(in_channels=64, out_channels=1, kernel_size=3)
        >>> x = torch.randn(2, 64, 128)  
        >>> out = head(x)
        >>> print(out.shape)
        torch.Size([2, 1, 128])
    """
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        return self.conv(x)


class UNetHead2D(nn.Module):
    """
    二维 U-Net 预测头 (UNet Head 2D)。

    将解码器输出的二维特征图映射到最终的目标张量。
    默认使用 1x1 卷积。若增大 kernel_size，会自动 padding 保证分辨率对齐。

    Args:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        kernel_size (int, optional): 预测头卷积核大小。默认值: 1。

    形状:
        输入 x: (B, C_in, H, W)
        输出: (B, C_out, H, W)

    Example:
        >>> head = UNetHead2D(in_channels=32, out_channels=3)
        >>> x = torch.randn(4, 32, 64, 64) 
        >>> out = head(x)
        >>> print(out.shape)
        torch.Size([4, 3, 64, 64])
    """
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        return self.conv(x)


class UNetHead3D(nn.Module):
    """
    三维 U-Net 预测头 (UNet Head 3D)。

    将解码器输出的三维特征映射到最终的体数据预测目标。
    支持灵活的 kernel_size 配置。

    Args:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        kernel_size (int, optional): 预测头卷积核大小。默认值: 1。

    形状:
        输入 x: (B, C_in, D, H, W)
        输出: (B, C_out, D, H, W)

    Example:
        >>> head = UNetHead3D(in_channels=16, out_channels=2, kernel_size=1)
        >>> x = torch.randn(1, 16, 8, 32, 32)  
        >>> out = head(x)
        >>> print(out.shape)
        torch.Size([1, 2, 8, 32, 32])
    """
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        return self.conv(x)