import torch
import torch.nn as nn
from onescience.modules import OneEncoder, OneDecoder, OneHead

class UNet(nn.Module):
    """
    该 U-Net 模型使用高度封装的编码器、解码器和预测头构建。
    - OneEncoder: 自动执行多级下采样，并返回所有层级的特征列表。
    - OneDecoder: 接收特征列表，自动匹配跳跃连接 (Skip Connections) 进行上采样融合。
    - OneHead: 将解码后的深层特征映射为目标物理量。

    Args:
        in_channels (int): 输入特征/图像的通道数。
        out_channels (int): 输出预测场的通道数。
        base_channels (int, optional): 初始特征通道数 (等价于原版的 features[0])。默认值: 16。
        num_stages (int, optional): 下采样/上采样的层数 (等价于原版 len(features)-1)。默认值: 2。
        bilinear (bool, optional): 是否使用双线性插值进行上采样。默认值: True。
        normtype (str, optional): 归一化类型 ('bn' 或 'in')。默认值: 'bn'。
        kernel_size (int, optional): 骨干网络的卷积核大小，必须为奇数。默认值: 3。

    形状:
        输入 x: (B, in_channels, H, W)
        输出 logits: (B, out_channels, H, W)

    Example:
        >>> model = UNet(in_channels=1, out_channels=2, base_channels=16, num_stages=2, kernel_size=5)
        >>> x = torch.randn(2, 1, 64, 64)
        >>> out = model(x)
        >>> print(out.shape)
        torch.Size([2, 2, 64, 64])
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
        super(UNet, self).__init__()
        
        self.encoder = OneEncoder(
            style="UNetEncoder2D",
            in_channels=in_channels,
            base_channels=base_channels,
            num_stages=num_stages,
            bilinear=bilinear,
            normtype=normtype,
            kernel_size=kernel_size
        )
        
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
            out_channels=out_channels,
            kernel_size=1  
        )

    def forward(self, x):
        features = self.encoder(x)
        decoded = self.decoder(features)
        logits = self.head(decoded)
        
        return logits