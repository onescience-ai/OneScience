import torch
from torch import nn
from torch.nn import functional as F
from timm.layers.helpers import to_2tuple
from timm.models.swin_transformer_v2 import SwinTransformerV2Stage
from onescience.modules import (
   OneEmbedding,
   OneTransformer,
   OneFc
)


class Fuxi(nn.Module):
    def __init__(self, 
                img_size=(2, 721, 1440), 
                patch_size=(2, 4, 4), 
                in_chans=70, 
                out_chans=70,
                embed_dim=1536, 
                num_groups=32, 
                num_heads=8, 
                window_size=7):
        super().__init__()
        input_resolution = int(img_size[1] / patch_size[1] / 2), int(img_size[2] / patch_size[2] / 2)
        self.patch_size = patch_size
        self.input_resolution = input_resolution
        self.out_chans = out_chans
        self.img_size = img_size

        self.cube_embedding = OneEmbedding(style="FuXiCubeEmbedding")
        self.u_transformer = OneTransformer(style="FuXiTransformer")
        self.fc = OneFc(style='FuXiFC')

    def forward(self, x: torch.Tensor):
        B, _, _, _, _ = x.shape
        _, patch_lat, patch_lon = self.patch_size
        Lat, Lon = self.input_resolution
        Lat, Lon = Lat * 2, Lon * 2
        x = self.cube_embedding(x).squeeze(2)  # B C Lat Lon
        x = self.u_transformer(x)
        x = self.fc(x.permute(0, 2, 3, 1))  # B Lat Lon C
        x = x.reshape(B, Lat, Lon, patch_lat, patch_lon, self.out_chans).permute(0, 1, 3, 2, 4, 5)
        # B, lat, patch_lat, lon, patch_lon, C

        x = x.reshape(B, Lat * patch_lat, Lon * patch_lon, self.out_chans)
        x = x.permute(0, 3, 1, 2)  # B C Lat Lon

        # bilinear
        x = F.interpolate(x, size=self.img_size[1:], mode="bilinear")

        return x