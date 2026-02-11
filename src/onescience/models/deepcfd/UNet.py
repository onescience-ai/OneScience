import torch
import torch.nn as nn
import torch.nn.functional as F
import onescience.modules.layer.layers import DoubleConv2D, Down2D, Up2D, OutConv2D

class UNet(nn.Module):
    """
    基于模块化组件重构的 U-Net 模型。

    该模型由编码器（下采样路径）、瓶颈层和解码器（上采样路径）组成。
    编码器逐步降低特征图的空间分辨率并增加通道数。
    解码器逐步恢复空间分辨率，并通过跳跃连接（Skip Connections）融合编码器的高分辨率特征。

    Args:
        in_channels (int): 输入图像的通道数。
        out_channels (int): 输出图像的通道数。
        features (list[int]): 每个层级的特征通道数列表。默认值: [16, 32, 64]。
        normtype (str): 归一化类型 ('bn' 或 'in')。默认值: 'bn'。
    """
    def __init__(self, in_channels, out_channels, features=[16, 32, 64], normtype="bn"):
        super(UNet, self).__init__()
        
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # 显式定义池化层，配合 Down2D 使用或单独使用

        # --- Encoder Path ---
        # 初始双卷积层 (不进行下采样)
        self.inc = DoubleConv2D(in_channels, features[0], normtype=normtype)
        
        # 下采样层
        for i in range(len(features) - 1):
            # Down2D 包含 MaxPool + DoubleConv
            self.encoders.append(
                Down2D(features[i], features[i+1], normtype=normtype)
            )
        for i in range(len(features) - 1, 0, -1):
            self.decoders.append(
                Up2D(features[i] + features[i-1], features[i-1], bilinear=True, normtype=normtype)
            )

        # --- Output Path ---
        self.outc = OutConv2D(features[0], out_channels)

    def forward(self, x):
        # 存储跳跃连接的特征
        skips = []
        
        # 初始卷积
        x = self.inc(x)
        skips.append(x)
        
        # 编码器路径
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
        
        x = skips.pop() 
        
        # 解码器路径
        for decoder in self.decoders:
            skip = skips.pop() # 获取对应的跳跃连接特征
            x = decoder(x, skip) # x 是深层特征，skip 是浅层特征
            
        # 输出层
        logits = self.outc(x)
        return logits