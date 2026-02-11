import torch
import torch.nn as nn
import torch.nn.functional as F
import onescience.modules.layer.layers import DoubleConv2D, Down2D, Up2D, OutConv2D

class DecoderPath(nn.Module):
    """
    单个解码器路径辅助类。
    
    用于 UNetEx 中。它接收瓶颈层特征和跳跃连接，逐步上采样并输出单通道结果。
    """
    def __init__(self, features, normtype="bn"):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # features 顺序例如 [16, 32, 64]
        # 解码过程是倒序的 (Deep -> Shallow): 64 -> 32 -> 16
        for i in range(len(features) - 1, 0, -1):
            # Up2D(in_channels, out_channels, ...)
            # in_channels = current_deep_channels + skip_connection_channels
            # out_channels = next_shallow_channels
            self.layers.append(
                Up2D(features[i] + features[i-1], features[i-1], bilinear=True, normtype=normtype)
            )
        
        # 按照原 UNetEx 逻辑，每个解码器头输出 1 个通道
        self.outc = OutConv2D(features[0], 1)

    def forward(self, x, skips):
        """
        Args:
            x: 瓶颈层特征 (Bottleneck feature)
            skips: 编码器产生的跳跃连接列表 [feat0, feat1, ...]
        """
        # 跳跃连接列表是 [浅 -> 深]，解码需要 [深 -> 浅]
        # 我们使用索引倒序访问，避免使用 pop() 破坏列表，从而允许被多个解码器复用
        skip_idx = len(skips) - 1
        
        for layer in self.layers:
            skip = skips[skip_idx]
            x = layer(x, skip)
            skip_idx -= 1
            
        return self.outc(x)


class UNetEx(nn.Module):
    """
    基于模块化组件重构的 UNetEx 模型。

    特点：
    1. **共享编码器**: 提取通用的图像/物理场特征。
    2. **多头解码器**: 针对 `out_channels` 中的每一个通道，都有一个完全独立的解码路径。
       最后将所有解码器的输出在通道维度拼接。

    Args:
        in_channels (int): 输入通道数。
        out_channels (int): 总输出通道数（决定了解码器头的数量）。
        features (list[int]): 特征通道列表。
        normtype (str): 归一化类型。
    """
    def __init__(self, in_channels, out_channels, features=[16, 32, 64], normtype="bn"):
        super(UNetEx, self).__init__()
        
        # --- Encoder Path (Shared) ---
        self.encoders = nn.ModuleList()
        # 初始层
        self.inc = DoubleConv2D(in_channels, features[0], normtype=normtype)
        
        # 下采样层
        for i in range(len(features) - 1):
            self.encoders.append(
                Down2D(features[i], features[i+1], normtype=normtype)
            )
            
        # --- Decoder Paths (Multiple Independent Heads) ---
        # 为每个输出通道创建一个独立的解码路径
        self.decoders = nn.ModuleList()
        for _ in range(out_channels):
            self.decoders.append(DecoderPath(features, normtype=normtype))

    def forward(self, x):
        # --- Encode (Shared) ---
        skips = []
        x = self.inc(x)
        skips.append(x)
        
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            
        # 此时 x 是最深层的特征 (Bottleneck)
        # skips 包含了所有层的特征 [L0, L1, ..., Bottleneck]
        # 我们需要把 Bottleneck 拿出来作为输入，剩下的作为跳跃连接
        bottleneck = skips.pop() 
        
        # --- Decode (Parallel/Loop) ---
        outputs = []
        for decoder in self.decoders:
            # 每个解码器复用相同的 bottleneck 和 skips
            out = decoder(bottleneck, skips)
            outputs.append(out)
            
        # 拼接所有头的输出 (B, 1, H, W) -> (B, out_channels, H, W)
        return torch.cat(outputs, dim=1)