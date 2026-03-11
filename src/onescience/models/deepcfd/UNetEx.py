import torch
import torch.nn as nn
from onescience.modules import OneEncoder, OneDecoder, OneHead

class DecoderPath(nn.Module):
    """
    辅助类：单通道独立解码器路径。
    包含一个 OneDecoder (负责上采样和特征融合) 和一个 OneHead (负责映射到单通道输出)。

    Args:
        base_channels (int): 初始特征通道数。
        num_stages (int): 上采样的层数。
        bilinear (bool): 是否使用双线性插值。
        normtype (str): 归一化类型 ('bn' 或 'in')。
        kernel_size (int): 解码器中的卷积核大小。

    形状:
        输入 features: list of Tensors (编码器输出的特征金字塔)。
        输出: (B, 1, H, W)
    """
    def __init__(self, base_channels, num_stages, bilinear, normtype, kernel_size):
        super().__init__()
        self.decoder = OneDecoder(
            style="UNetDecoder2D",
            base_channels=base_channels,
            num_stages=num_stages,
            bilinear=bilinear,
            normtype=normtype,
            kernel_size=kernel_size
        )
        self.head = OneHead(
            style="UNetHead2D",
            in_channels=base_channels,
            out_channels=1,
            kernel_size=1
        )

    def forward(self, features):
        decoded = self.decoder(features)
        return self.head(decoded)


class UNetEx(nn.Module):
    """
    基于模块化组件工厂构建的多头 U-Net 模型 (UNetEx)。

    特点：
    1. 共享编码器 (Shared Encoder): 提取通用的物理场/图像特征。
    2. 多头解码器 (Independent Decoders): 为每一个输出通道构建一条完全独立的解码和跳跃连接路径。
       所有解码路径的输出最后在通道维度拼接。

    Args:
        in_channels (int): 输入图像/物理场的通道数。
        out_channels (int): 总输出通道数（决定了独立解码器头的数量）。
        base_channels (int, optional): 初始特征通道数。默认值: 16。
        num_stages (int, optional): 下采样/上采样的层数。默认值: 2。
        bilinear (bool, optional): 是否使用双线性插值进行上采样。默认值: True。
        normtype (str, optional): 归一化类型 ('bn' 或 'in')。默认值: 'bn'。
        kernel_size (int, optional): 骨干网络的卷积核大小，必须为奇数。默认值: 3。

    形状:
        输入 x: (B, in_channels, H, W)
        输出: (B, out_channels, H, W)

    Example:
        >>> model = UNetEx(in_channels=3, out_channels=4, base_channels=16, num_stages=2, kernel_size=3)
        >>> x = torch.randn(2, 3, 64, 64)
        >>> out = model(x)
        >>> print(out.shape)
        torch.Size([2, 4, 64, 64])
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        base_channels: int = 16, 
        num_stages: int = 2, 
        bilinear: bool = True, 
        normtype: str = "bn",
        kernel_size: int = 3
    ):
        super(UNetEx, self).__init__()
        
        self.encoder = OneEncoder(
            style="UNetEncoder2D",
            in_channels=in_channels,
            base_channels=base_channels,
            num_stages=num_stages,
            bilinear=bilinear,
            normtype=normtype,
            kernel_size=kernel_size
        )
            
        self.decoders = nn.ModuleList([
            DecoderPath(base_channels, num_stages, bilinear, normtype, kernel_size)
            for _ in range(out_channels)
        ])

    def forward(self, x):
        features = self.encoder(x)
        
        outputs = []
        for decoder_path in self.decoders:
            out = decoder_path(features)
            outputs.append(out)
            
        return torch.cat(outputs, dim=1)