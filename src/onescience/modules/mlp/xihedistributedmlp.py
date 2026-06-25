import torch
import torch.nn as nn
from onescience.modules.func_utils import Mlp, DistributedMlp


class XiheDistributedMlp(nn.Module):
    def __init__(
        self,
        dim,
        num_groups=32,
        mlp_ratio=4.0,
        drop=0.0,
        act_layer=nn.GELU,
        LN=nn.LayerNorm,
        config=None,
    ):
        super().__init__()
        self.dim = dim
        self.num_groups = num_groups

        self.norm1 = LN(dim)
        self.norm2 = LN(dim)

        mlp_token_dim = int(num_groups * mlp_ratio)
        self.mlp_token = Mlp(
            in_features=num_groups,
            hidden_features=mlp_token_dim,
            act_layer=act_layer,
            drop=drop,
        )

        mlp_channel_dim = int(dim * mlp_ratio)
        self.mlp_channel = DistributedMlp(
            in_features=dim,
            hidden_features=mlp_channel_dim,
            act_layer=act_layer,
            drop=drop,
            config=config,
        )

    def forward(self, x):
        B, G, C = x.shape
        shortcut = x

        x = self.norm1(x)
        x = x.transpose(1, 2)
        x = self.mlp_token(x)
        x = x.transpose(1, 2)
        x = shortcut + x

        y = self.norm2(x)
        y = self.mlp_channel(y)
        y = x + y
        return y
