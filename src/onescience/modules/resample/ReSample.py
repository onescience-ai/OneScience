# modules/resample.py
import torch
from torch import nn
import warnings


class OneReSample:
    """
    统一的空间重采样接口，支持多种实现风格。
    
    通过 style 参数选择不同模型的重采样实现策略，所有风格共享相同的接口。
    
    Args:
        in_dim (int): 输入通道数
        out_dim (int): 输出通道数
        input_resolution (tuple): 输入空间维度，2元组(H,W)或3元组(P,H,W)
        output_resolution (tuple): 输出空间维度，形状需与input_resolution匹配
        style (str): 重采样实现风格，默认'pangu'
            可选值: ['pangu']
        **kwargs: 各style特定参数（当前pangu不需要额外参数）
    
    Examples:
        >>> # 使用pangu风格（默认）
        >>> resample = OneReSample(128, 64, (64, 128), (128, 256))
        >>> resample = OneReSample(128, 64, (64, 128), (128, 256), style='pangu')
        
        >>> # 未来可扩展其他风格
    """
    
    _registry = {}
    
    def __new__(cls, in_dim, out_dim, input_resolution, output_resolution, 
                style='pangu', **kwargs):
        if style not in cls._registry:
            available_styles = list(cls._registry.keys())
            warnings.warn(
                f"Style '{style}' not available. Available styles: {available_styles}. "
                f"Using 'pangu' as fallback.",
                UserWarning
            )
            style = 'pangu'
        
        return cls._registry[style](
            in_dim, out_dim, input_resolution, output_resolution, **kwargs
        )
    
    @classmethod
    def register(cls, name):
        def wrapper(resample_class):
            cls._registry[name] = resample_class
            return resample_class
        return wrapper
    
    @classmethod
    def list_styles(cls):
        return list(cls._registry.keys())

        
@OneReSample.register('pangu')
class PanguReSample(nn.Module):
    """
    Pangu-Weather风格的空间重采样实现。
    
    改编自 WeatherLearn 项目和 Pangu-Weather 官方实现。
    可根据输入和目标分辨率自动判断上采样或下采样，支持2D/3D数据。
    
    Args:
        in_dim (int): 输入通道数
        out_dim (int): 输出通道数
        input_resolution (tuple): 输入空间维度
            - 2D: (H, W)
            - 3D: (P, H, W)
        output_resolution (tuple): （上或下采样后的）输出空间维度
            - 2D: (H', W')
            - 3D: (P', H', W')
    
    形状:
        - 2D输入: (B, H×W, C_in) -> (B, H'×W', C_out)
        - 3D输入: (B, P×H×W, C_in) -> (B, P'×H'×W', C_out)
    
    Examples:
        >>> # 2D上采样 - 对应原UpSample2D
        >>> resample = OneReSample(128, 64, (64, 128), (128, 256), style='pangu')
        >>> x = torch.randn(8, 8192, 128)
        >>> out = resample(x)
        >>> out.shape
        torch.Size([8, 32768, 64])
        
        >>> # 3D上采样 - 对应原UpSample3D
        >>> resample = OneReSample(256, 128, (13, 32, 64), (13, 64, 128), style='pangu')
        >>> x = torch.randn(4, 26624, 256)
        >>> out = resample(x)
        >>> out.shape
        torch.Size([4, 106496, 128])
        
        >>> # 2D下采样 - 对应原DownSample2D
        >>> resample = OneReSample(64, 128, (128, 256), (64, 128), style='pangu')
        >>> x = torch.randn(8, 32768, 64)
        >>> out = resample(x)
        >>> out.shape
        torch.Size([8, 8192, 128])
        
        >>> # 3D下采样 - 对应原DownSample3D
        >>> resample = OneReSample(128, 256, (13, 64, 128), (13, 32, 64), style='pangu')
        >>> x = torch.randn(4, 106496, 128)
        >>> out = resample(x)
        >>> out.shape
        torch.Size([4, 26624, 256])
    """
    
    def __init__(self, in_dim, out_dim, input_resolution, output_resolution, **kwargs):
        super().__init__()
        
        # 参数验证（可选：kwargs验证）
        if kwargs:
            warnings.warn(
                f"PanguReSample received unexpected kwargs: {list(kwargs.keys())}. "
                f"These will be ignored.",
                UserWarning
            )
        
        # [这里是原来ReSample的完整实现代码]
        # 验证分辨率维度匹配性
        if len(input_resolution) != len(output_resolution):
            raise ValueError(
                f"Resolution shape mismatch: input_resolution has "
                f"{len(input_resolution)} dimensions ({input_resolution}), but output_resolution has "
                f"{len(output_resolution)} dimensions ({output_resolution})"
            )
        
        if len(input_resolution) not in [2, 3]:
            raise ValueError(
                f"input_resolution only support 2D or 3D resolution, got {len(input_resolution)}D: {input_resolution}"
            )
        if len(output_resolution) not in [2, 3]:
            raise ValueError(
                f"output_resolution only support 2D or 3D resolution, got {len(output_resolution)}D: {output_resolution}"
            )

        
        if len(input_resolution) == 2:
            self.input_resolution = (1,) + tuple(input_resolution)
            self.output_resolution = (1,) + tuple(output_resolution)
        else:
            self.input_resolution = tuple(input_resolution)
            self.output_resolution = tuple(output_resolution)
        
        input_size = self.input_resolution[0] * self.input_resolution[1] * self.input_resolution[2]
        output_size = self.output_resolution[0] * self.output_resolution[1] * self.output_resolution[2]
        
        if input_size == output_size:
            raise ValueError(
                f"Input and output resolutions are identical ({input_resolution} -> {output_resolution}). "
                f"ReSample requires different resolutions for up/downsampling."
            )
        
        self.is_upsample = output_size > input_size
        
        if self.is_upsample:
            self._init_upsample(in_dim, out_dim)
        else:
            self._init_downsample(in_dim, out_dim)
    
    def _init_upsample(self, in_dim, out_dim):
        self.linear1 = nn.Linear(in_dim, out_dim * 4, bias=False)
        self.linear2 = nn.Linear(out_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
    
    def _init_downsample(self, in_dim, out_dim):
        self.linear = nn.Linear(in_dim * 4, out_dim, bias=False)
        self.norm = nn.LayerNorm(4 * in_dim)
        
        in_pl, in_lat, in_lon = self.input_resolution
        out_pl, out_lat, out_lon = self.output_resolution
        
        h_pad = out_lat * 2 - in_lat
        w_pad = out_lon * 2 - in_lon
        
        pad_top = h_pad // 2
        pad_bottom = h_pad - pad_top
        pad_left = w_pad // 2
        pad_right = w_pad - pad_left
        
        self.pad = nn.ZeroPad3d(
            (pad_left, pad_right, pad_top, pad_bottom, 0, 0)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_upsample:
            return self._forward_upsample(x)
        else:
            return self._forward_downsample(x)
    
    def _forward_upsample(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        in_pl, in_lat, in_lon = self.input_resolution
        out_pl, out_lat, out_lon = self.output_resolution
        
        x = self.linear1(x)
        x = x.reshape(B, in_pl, in_lat, in_lon, 2, 2, C // 2).permute(0, 1, 2, 4, 3, 5, 6)
        x = x.reshape(B, in_pl, in_lat * 2, in_lon * 2, -1)
        
        pad_h = in_lat * 2 - out_lat
        pad_w = in_lon * 2 - out_lon
        
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        x = x[:, :out_pl, pad_top : 2 * in_lat - pad_bottom, pad_left : 2 * in_lon - pad_right, :]
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], x.shape[4])
        
        x = self.norm(x)
        x = self.linear2(x)
        
        return x
    
    def _forward_downsample(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        in_pl, in_lat, in_lon = self.input_resolution
        out_pl, out_lat, out_lon = self.output_resolution
        
        x = x.reshape(B, in_pl, in_lat, in_lon, C)
        x = self.pad(x.permute(0, -1, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        x = x.reshape(B, in_pl, out_lat, 2, out_lon, 2, C).permute(0, 1, 2, 4, 3, 5, 6)
        x = x.reshape(B, out_pl * out_lat * out_lon, 4 * C)
        
        x = self.norm(x)
        x = self.linear(x)
        
        return x