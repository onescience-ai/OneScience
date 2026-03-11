import torch
import torch.nn as nn
import torch.nn.functional as F

################################################################
# Multiscale modules 1D
################################################################

class DoubleConv1D(nn.Module):
    """
    一维双卷积块 (1D Double Convolution Block)。

    该模块包含两个连续的卷积层，每个卷积层后接归一化层 (BatchNorm/InstanceNorm) 和 ReLU 激活函数。
    结构为：(Conv1d => [BN/IN] => ReLU) * 2。这是 U-Net 中提取特征的核心组件。

    Args:
        in_channels (int): 输入张量的通道数。
        out_channels (int): 输出张量的通道数。
        mid_channels (int, optional): 中间层的通道数。如果为 None，则默认等于 out_channels。
        normtype (str, optional): 归一化类型。支持 'bn' (BatchNorm), 'in' (InstanceNorm)。默认为 'bn'。
        kernel_size (int, optional): 卷积核大小。必须为奇数以保证 padding 对齐。默认为 3。

    形状:
        输入: (B, C_in, L)
        输出: (B, C_out, L)

    Example:
        >>> dconv = DoubleConv1D(64, 32, kernel_size=5)
        >>> x = torch.randn(8, 64, 100)
        >>> out = dconv(x)
        >>> print(out.shape)
        torch.Size([8, 32, 100])
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, normtype="bn", kernel_size=3):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        assert kernel_size % 2 == 1, "kernel_size must be odd to maintain spatial dimensions."
        padding = kernel_size // 2

        if normtype == "bn":
            self.double_conv = nn.Sequential(
                nn.Conv1d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=False),
                nn.BatchNorm1d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
            )
        elif normtype == "in":
            self.double_conv = nn.Sequential(
                nn.Conv1d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=False),
                nn.InstanceNorm1d(mid_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv1d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
                nn.InstanceNorm1d(out_channels, affine=True),
                nn.ReLU(inplace=True),
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv1d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv1d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.double_conv(x)


class Down1D(nn.Module):
    """
    一维下采样模块 (1D Downsampling Block)。

    首先通过最大池化层（MaxPool1d）将空间维度减半，然后应用 DoubleConv1D 提取特征。
    用于 U-Net 的编码器路径。

    Args:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        normtype (str, optional): 归一化类型 ('bn' 或 'in')。默认为 'bn'。
        kernel_size (int, optional): 卷积核大小。默认为 3。

    形状:
        输入: (B, C_in, L)
        输出: (B, C_out, L / 2)

    Example:
        >>> down = Down1D(64, 128, kernel_size=3)
        >>> x = torch.randn(8, 64, 100)
        >>> out = down(x)
        >>> print(out.shape)
        torch.Size([8, 128, 50])
    """
    def __init__(self, in_channels, out_channels, normtype="bn", kernel_size=3):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2), 
            DoubleConv1D(in_channels, out_channels, normtype=normtype, kernel_size=kernel_size)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up1D(nn.Module):
    """
    一维上采样模块 (1D Upsampling Block)。

    对输入 x1 进行上采样，然后与来自编码器路径的跳跃连接特征 x2 进行拼接 (Concatenate)，
    最后通过 DoubleConv1D 融合特征。用于 U-Net 的解码器路径。

    Args:
        in_channels (int): 拼接后的总输入通道数。
        out_channels (int): 输出通道数。
        bilinear (bool, optional): 上采样方式。True 为线性插值，False 为转置卷积。默认为 True。
        normtype (str, optional): 归一化类型。默认为 'bn'。
        kernel_size (int, optional): 融合卷积的卷积核大小。默认为 3。

    形状:
        输入 x1 (深层特征): (B, C_1, L)
        输入 x2 (跳跃连接): (B, C_2, 2L)
        输出: (B, C_out, 2L)

    Example:
        >>> up = Up1D(128, 64, bilinear=True)
        >>> x1 = torch.randn(8, 64, 50) 
        >>> x2 = torch.randn(8, 64, 100)
        >>> out = up(x1, x2)
        >>> print(out.shape)
        torch.Size([8, 64, 100])
    """
    def __init__(self, in_channels, out_channels, bilinear=True, normtype="bn", kernel_size=3):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
            self.conv = DoubleConv1D(in_channels, out_channels, in_channels // 2, normtype=normtype, kernel_size=kernel_size)
        else:
            self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv1D(in_channels, out_channels, normtype=normtype, kernel_size=kernel_size)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv1D(nn.Module):
    """
    一维输出卷积层 (1D Output Convolution)。

    使用 1x1 卷积将特征图映射到最终的输出通道（如物理场变量维度）。

    Args:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。

    形状:
        输入: (B, C_in, L)
        输出: (B, C_out, L)

    Example:
        >>> out_conv = OutConv1D(64, 10)
        >>> x = torch.randn(8, 64, 100)
        >>> out = out_conv(x)
        >>> print(out.shape)
        torch.Size([8, 10, 100])
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv1D, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


################################################################
# Multiscale modules 2D
################################################################

class DoubleConv2D(nn.Module):
    """
    二维双卷积块 (2D Double Convolution Block)。

    结构为：(Conv2d => [BN/IN] => ReLU) * 2。
    支持通过 kernel_size 参数动态调整感受野，同时自动计算 padding 以维持空间尺寸。

    Args:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        mid_channels (int, optional): 中间层通道数。默认为 None (即等于 out_channels)。
        normtype (str, optional): 归一化类型 ('bn', 'in')。默认为 'bn'。
        kernel_size (int, optional): 卷积核大小，必须为奇数。默认为 3。

    形状:
        输入: (B, C_in, H, W)
        输出: (B, C_out, H, W)

    Example:
        >>> dconv = DoubleConv2D(64, 32, kernel_size=5)
        >>> x = torch.randn(8, 64, 128, 128)
        >>> out = dconv(x)
        >>> print(out.shape)
        torch.Size([8, 32, 128, 128])
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, normtype="bn", kernel_size=3):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        assert kernel_size % 2 == 1, "kernel_size must be odd to maintain spatial dimensions."
        padding = kernel_size // 2

        if normtype == "bn":
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        elif normtype == "in":
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=False),
                nn.InstanceNorm2d(mid_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
                nn.InstanceNorm2d(out_channels, affine=True),
                nn.ReLU(inplace=True),
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.double_conv(x)


class Down2D(nn.Module):
    """
    二维下采样模块 (2D Downsampling Block)。

    使用 2x2 最大池化将高宽减半，随后连接 DoubleConv2D 提取特征。

    Args:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        normtype (str, optional): 归一化类型。默认为 'bn'。
        kernel_size (int, optional): 卷积核大小。默认为 3。

    形状:
        输入: (B, C_in, H, W)
        输出: (B, C_out, H/2, W/2)

    Example:
        >>> down = Down2D(64, 128)
        >>> x = torch.randn(8, 64, 128, 128)
        >>> out = down(x)
        >>> print(out.shape)
        torch.Size([8, 128, 64, 64])
    """
    def __init__(self, in_channels, out_channels, normtype="bn", kernel_size=3):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), 
            DoubleConv2D(in_channels, out_channels, normtype=normtype, kernel_size=kernel_size)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up2D(nn.Module):
    """
    二维上采样模块 (2D Upsampling Block)。

    对输入 x1 进行上采样，对齐尺寸后与跳跃连接 x2 拼接，再经过卷积融合。
    内置自动 Padding 逻辑，可健壮处理 x1 和 x2 因池化奇偶性导致的轻微尺寸不匹配。

    Args:
        in_channels (int): 拼接后的总输入通道数。
        out_channels (int): 输出通道数。
        bilinear (bool, optional): 是否使用双线性插值上采样。默认为 True。
        normtype (str, optional): 归一化类型。默认为 'bn'。
        kernel_size (int, optional): 融合卷积核大小。默认为 3。

    形状:
        输入 x1: (B, C_1, H, W)
        输入 x2: (B, C_2, H_target, W_target)
        输出: (B, C_out, H_target, W_target)

    Example:
        >>> up = Up2D(128, 64, bilinear=True)
        >>> x1 = torch.randn(8, 64, 64, 64)
        >>> x2 = torch.randn(8, 64, 128, 128)
        >>> out = up(x1, x2)
        >>> print(out.shape)
        torch.Size([8, 64, 128, 128])
    """
    def __init__(self, in_channels, out_channels, bilinear=True, normtype="bn", kernel_size=3):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv2D(in_channels, out_channels, in_channels // 2, normtype=normtype, kernel_size=kernel_size)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv2D(in_channels, out_channels, normtype=normtype, kernel_size=kernel_size)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv2D(nn.Module):
    """
    二维输出卷积层 (2D Output Convolution)。

    使用 1x1 卷积进行通道映射。

    Args:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。

    形状:
        输入: (B, C_in, H, W)
        输出: (B, C_out, H, W)

    Example:
        >>> out_conv = OutConv2D(64, 3) 
        >>> x = torch.randn(8, 64, 128, 128)
        >>> out = out_conv(x)
        >>> print(out.shape)
        torch.Size([8, 3, 128, 128])
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


################################################################
# Multiscale modules 3D
################################################################

class DoubleConv3D(nn.Module):
    """
    三维双卷积块 (3D Double Convolution Block)。

    结构为：(Conv3d => [BN/IN] => ReLU) * 2。
    用于处理体素数据或时空演化数据，支持通过 kernel_size 参数调整三维感受野。

    Args:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        mid_channels (int, optional): 中间通道数。默认为 None。
        normtype (str, optional): 归一化类型 ('bn', 'in')。默认为 'bn'。
        kernel_size (int, optional): 卷积核大小，必须为奇数。默认为 3。

    形状:
        输入: (B, C_in, D, H, W)
        输出: (B, C_out, D, H, W)

    Example:
        >>> dconv = DoubleConv3D(64, 32)
        >>> x = torch.randn(4, 64, 16, 32, 32)
        >>> out = dconv(x)
        >>> print(out.shape)
        torch.Size([4, 32, 16, 32, 32])
    """
    def __init__(self, in_channels, out_channels, mid_channels=None, normtype="bn", kernel_size=3):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        assert kernel_size % 2 == 1, "kernel_size must be odd to maintain spatial dimensions."
        padding = kernel_size // 2

        if normtype == "bn":
            self.double_conv = nn.Sequential(
                nn.Conv3d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=False),
                nn.BatchNorm3d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
            )
        elif normtype == "in":
            self.double_conv = nn.Sequential(
                nn.Conv3d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=False),
                nn.InstanceNorm3d(mid_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True),
                nn.ReLU(inplace=True),
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv3d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv3d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.double_conv(x)


class Down3D(nn.Module):
    """
    三维下采样模块 (3D Downsampling Block)。

    使用 2x2x2 最大池化将空间维度减半，然后进行双卷积特征提取。

    Args:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        normtype (str, optional): 归一化类型。默认为 'bn'。
        kernel_size (int, optional): 卷积核大小。默认为 3。

    形状:
        输入: (B, C_in, D, H, W)
        输出: (B, C_out, D/2, H/2, W/2)

    Example:
        >>> down = Down3D(32, 64)
        >>> x = torch.randn(4, 32, 16, 32, 32)
        >>> out = down(x)
        >>> print(out.shape)
        torch.Size([4, 64, 8, 16, 16])
    """
    def __init__(self, in_channels, out_channels, normtype="bn", kernel_size=3):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2), 
            DoubleConv3D(in_channels, out_channels, normtype=normtype, kernel_size=kernel_size)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    """
    三维上采样模块 (3D Upsampling Block)。

    对输入特征进行三维上采样，并与高分辨率的跳跃连接特征拼接，完成体数据解码。

    Args:
        in_channels (int): 拼接后的总输入通道数。
        out_channels (int): 输出通道数。
        bilinear (bool, optional): 是否使用三线性插值 (trilinear)。默认为 True。
        normtype (str, optional): 归一化类型。默认为 'bn'。
        kernel_size (int, optional): 融合卷积核大小。默认为 3。

    形状:
        输入 x1: (B, C_1, D, H, W)
        输入 x2: (B, C_2, 2D, 2H, 2W)
        输出: (B, C_out, 2D, 2H, 2W)

    Example:
        >>> up = Up3D(128, 64, bilinear=True)
        >>> x1 = torch.randn(4, 64, 8, 16, 16)
        >>> x2 = torch.randn(4, 64, 16, 32, 32)
        >>> out = up(x1, x2)
        >>> print(out.shape)
        torch.Size([4, 64, 16, 32, 32])
    """
    def __init__(self, in_channels, out_channels, bilinear=True, normtype="bn", kernel_size=3):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels, in_channels // 2, normtype=normtype, kernel_size=kernel_size)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels, normtype=normtype, kernel_size=kernel_size)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv3D(nn.Module):
    """
    三维输出卷积层 (3D Output Convolution)。

    使用 1x1x1 卷积将深层特征映射到最终的输出空间。

    Args:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。

    形状:
        输入: (B, C_in, D, H, W)
        输出: (B, C_out, D, H, W)

    Example:
        >>> out_conv = OutConv3D(64, 1) 
        >>> x = torch.randn(4, 64, 16, 32, 32)
        >>> out = out_conv(x)
        >>> print(out.shape)
        torch.Size([4, 1, 16, 32, 32])
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)