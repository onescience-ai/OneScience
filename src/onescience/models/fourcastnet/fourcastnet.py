from functools import partial
import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from einops import rearrange, repeat
from onescience.modules.fc.onefc import OneFC
from onescience.modules.afno.oneafno import OneAFNO
from onescience.modules.embedding.oneembedding import OneEmbedding
from onescience.modules.fuser.onefuser import OneFuser


class FourCastNet(nn.Module):
    def __init__(
            self,
            img_size=(720, 1440),
            patch_size=(8, 8),
            in_chans=19,
            out_chans=19,
            embed_dim=768,
            depth=12,
            mlp_ratio=4.,
            drop_rate=0.,
            drop_path_rate=0.,
            num_blocks=8,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0,
        ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.num_features = self.embed_dim = embed_dim
        self.num_blocks = num_blocks 
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.h = img_size[0] // self.patch_size[0]
        self.w = img_size[1] // self.patch_size[1]

        self.patch_embed = OneEmbedding(style="FourCastNetEmbedding")
        
        self.blocks = nn.ModuleList([
            OneFuser(style="FourCastNetFuser") 
            for i in range(depth)
            ]
        )

        self.head = nn.Linear(embed_dim, self.out_chans*self.patch_size[0]*self.patch_size[1], bias=False)

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}


    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        x = x.reshape(B, self.h, self.w, self.embed_dim)
        for blk in self.blocks:
            x = blk(x)

        x = self.head(x)
        x = rearrange(
            x,
            "b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            h=self.img_size[0] // self.patch_size[0],
            w=self.img_size[1] // self.patch_size[1],
        )
        return x