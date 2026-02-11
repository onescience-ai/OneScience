import torch
from torch import nn
from timm.layers.helpers import to_2tuple
from timm.models.swin_transformer_v2 import SwinTransformerV2Stage
import onescience
from onescience.modules.func_utils.fuxi_utils import get_pad2d


class FuXiTransformer(nn.Module):
    """U-Transformer
    Args:
        embed_dim (int): Patch embedding dimension.
        num_groups (int | tuple[int]): number of groups to separate the channels into.
        input_resolution (tuple[int]): Lat, Lon.
        num_heads (int): Number of attention heads in different layers.
        window_size (int | tuple[int]): Window size.
        depth (int): Number of blocks.
    """
    def __init__(self, 
                 embed_dim=1536, 
                 num_groups=32, 
                 input_resolution=(int(721 / 4 / 2), int(1440 / 4 / 2)),
                 num_heads=8, 
                 window_size=7, 
                 depth=48):
        super().__init__()
        num_groups = to_2tuple(num_groups)
        window_size = to_2tuple(window_size)
        padding = get_pad2d(input_resolution, window_size)
        padding_left, padding_right, padding_top, padding_bottom = padding
        self.padding = padding
        self.pad = nn.ZeroPad2d(padding)
        input_resolution = list(input_resolution)
        input_resolution[0] = input_resolution[0] + padding_top + padding_bottom
        input_resolution[1] = input_resolution[1] + padding_left + padding_right
        self.down = onescience.modules.OneSample(style='FuXiDownSample')
        self.layer = SwinTransformerV2Stage(embed_dim, embed_dim, input_resolution, depth, num_heads, window_size)
        self.up = onescience.modules.OneSample(style='FuXiUpSample')

    def forward(self, x):
        B, C, Lat, Lon = x.shape
        padding_left, padding_right, padding_top, padding_bottom = self.padding
        
        x = self.down(x)

        shortcut = x

        # pad
        x = self.pad(x)
        _, _, pad_lat, pad_lon = x.shape

        x = x.permute(0, 2, 3, 1)  # B Lat Lon C
        x = self.layer(x)
        x = x.permute(0, 3, 1, 2)

        # crop
        x = x[:, :, padding_top: pad_lat - padding_bottom, padding_left: pad_lon - padding_right]

        # concat
        x = torch.cat([shortcut, x], dim=1)  # B 2*C Lat Lon

        x = self.up(x)
        return x

