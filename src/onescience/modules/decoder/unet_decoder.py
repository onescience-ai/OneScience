import torch
import torch.nn as nn
from onescience.modules.layer.unet_layer import Up1D, Up2D, Up3D

class UNetDecoder1D(nn.Module):
    """
    一维 U-Net 解码器 (UNet Decoder 1D)。

    接收来自编码器的特征列表。它通过 Up1D 模块逐层上采样瓶颈特征，并与编码器中对应的浅层特征进行跳跃连接融合。

    Args:
        base_channels (int, optional): 与编码器匹配的初始特征通道数。默认值: 64。
        num_stages (int, optional): 上采样的层数。需与编码器一致。默认值: 4。
        bilinear (bool, optional): 是否使用双线性插值进行上采样。默认值: True。
        normtype (str, optional): 归一化类型 ('bn' 或 'in')。默认值: 'bn'。

    形状:
        输入 features: list of Tensors (由 UNetEncoder1D 输出)。
        输出: (B, base_channels, L)，恢复到输入分辨率的深层特征。

    Example:
        >>> decoder = UNetDecoder1D(base_channels=64, num_stages=4)
        >>> # features 是 encoder 的输出
        >>> out = decoder(features)
        >>> out.shape
        torch.Size([8, 64, 128])
    """
    def __init__(self, base_channels=16, num_stages=2, bilinear=True, normtype="bn"):
        super().__init__()
        self.up_stages = nn.ModuleList()
        # 生成与 Encoder 对应的通道列表，例如 [16, 32, 64]
        features = [base_channels * (2 ** i) for i in range(num_stages + 1)]
        
        for i in range(num_stages, 0, -1):
            in_ch = features[i] + features[i-1] # 深层 + 浅层
            out_ch = features[i-1]
            self.up_stages.append(Up1D(in_ch, out_ch, bilinear, normtype))

    def forward(self, features):
        x = features[-1]
        skips = reversed(features[:-1])
        for up_stage, skip in zip(self.up_stages, skips):
            x = up_stage(x, skip)
        return x


class UNetDecoder2D(nn.Module):
    """
    二维 U-Net 解码器 (UNet Decoder 2D)。

    接收来自二维编码器的特征列表，逐层进行 2D 上采样和跳跃融合，恢复空间分辨率。

    Args:
        base_channels (int, optional): 与编码器匹配的初始特征通道数。默认值: 64。
        num_stages (int, optional): 上采样的层数。需与编码器一致。默认值: 4。
        bilinear (bool, optional): 是否使用双线性插值进行上采样。默认值: True。
        normtype (str, optional): 归一化类型 ('bn' 或 'in')。默认值: 'bn'。

    形状:
        输入 features: list of Tensors (由 UNetEncoder2D 输出)。
        输出: (B, base_channels, H, W)。
    """
    def __init__(self, base_channels=16, num_stages=2, bilinear=True, normtype="bn"):
        super().__init__()
        self.up_stages = nn.ModuleList()
        features = [base_channels * (2 ** i) for i in range(num_stages + 1)]
        
        for i in range(num_stages, 0, -1):
            # --- 关键修改开始 ---
            if bilinear:
                in_ch = features[i] + features[i-1]
            else:
                in_ch = features[i]
            # --- 关键修改结束 ---
            
            out_ch = features[i-1]
            self.up_stages.append(Up2D(in_ch, out_ch, bilinear, normtype))

    def forward(self, features):
        x = features[-1]
        skips = reversed(features[:-1])
        for up_stage, skip in zip(self.up_stages, skips):
            x = up_stage(x, skip)
        return x


class UNetDecoder3D(nn.Module):
    """
    三维 U-Net 解码器 (UNet Decoder 3D)。

    接收来自三维编码器的特征列表，逐层进行 3D 上采样和跳跃融合，恢复体素/时空分辨率。

    Args:
        base_channels (int, optional): 与编码器匹配的初始特征通道数。默认值: 64。
        num_stages (int, optional): 上采样的层数。需与编码器一致。默认值: 4。
        bilinear (bool, optional): 是否使用三线性插值进行上采样。默认值: True。
        normtype (str, optional): 归一化类型 ('bn' 或 'in')。默认值: 'bn'。

    形状:
        输入 features: list of Tensors (由 UNetEncoder3D 输出)。
        输出: (B, base_channels, D, H, W)。
    """
    def __init__(self, base_channels=16, num_stages=2, bilinear=True, normtype="bn"):
        super().__init__()
        self.up_stages = nn.ModuleList()
        features = [base_channels * (2 ** i) for i in range(num_stages + 1)]
        
        for i in range(num_stages, 0, -1):
            in_ch = features[i] + features[i-1]
            out_ch = features[i-1]
            self.up_stages.append(Up3D(in_ch, out_ch, bilinear, normtype))

    def forward(self, features):
        x = features[-1]
        skips = reversed(features[:-1])
        for up_stage, skip in zip(self.up_stages, skips):
            x = up_stage(x, skip)
        return x