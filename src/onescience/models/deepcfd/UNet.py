import torch
import torch.nn as nn
from onescience.modules import OneEncoder, OneDecoder, OneHead

class UNet(nn.Module):
    """
    基于模块化组件工厂重构的 U-Net 模型。


    该模型彻底摒弃了底层算子的手动拼接，转而使用高度封装的编码器、解码器和预测头。
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
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        base_channels: int = 16, 
        num_stages: int = 2, 
        bilinear: bool = True,
        normtype: str = "bn"
    ):
        super(UNet, self).__init__()
        
        # 1. 实例化编码器 (负责提取特征并保留跳跃连接)
        self.encoder = OneEncoder(
            style="UNetEncoder2D",
            in_channels=in_channels,
            base_channels=base_channels,
            num_stages=num_stages,
            bilinear=bilinear,
            normtype=normtype
        )
        
        # 2. 实例化解码器 (负责接收列表并逐层上采样融合)
        self.decoder = OneDecoder(
            style="UNetDecoder2D",
            base_channels=base_channels,
            num_stages=num_stages,
            bilinear=bilinear,
            normtype=normtype
        )
        
        # 3. 实例化预测头 (负责输出通道映射)
        self.head = OneHead(
            style="UNetHead2D",
            in_channels=base_channels,
            out_channels=out_channels
        )

    def forward(self, x):
        # 仅仅三行代码，完成了整个 U-Net 的前向传播！
        
        # 1. 编码器提取多尺度特征 (返回特征列表 [x1, x2, x3...])
        features = self.encoder(x)
        
        # 2. 解码器自动处理特征列表并融合
        decoded = self.decoder(features)
        
        # 3. 输出头映射到目标通道
        logits = self.head(decoded)
        
        return logits