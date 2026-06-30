import torch
import torch.nn as nn
from onescience.modules.func_utils import Mlp
from onescience.modules.attention.oneattention import OneAttention
from onescience.modules.mlp.onemlp import OneMlp


class XiheDistributedGlobalSIEFuser(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=12,
        qkv_bias=True,
        num_groups=32,
        norm_layer=nn.LayerNorm,
        config=None,
    ):
        super().__init__()
        self.dim = dim

        self.feature_grouping = OneAttention(
            dim=dim,
            num_heads=num_heads,
            num_groups=num_groups,
            style="DistributedFeatureGroupingAttention",
            config=config,
        )
        self.feature_ungrouping = OneAttention(
            dim=dim,
            num_heads=num_heads,
            style="DistributedFeatureUngroupingAttention",
            config=config,
        )

        self.group_propagation = OneMlp(
            dim=dim,
            num_groups=num_groups,
            style="XiheDistributedMlp",
            config=config,
        )

    def forward(self, obj):
        if isinstance(obj, dict):
            x = obj["x"]
            mask = obj.get("mask")
            if mask is not None:
                mask = mask.clone().detach().float()
            obj["y"] = x
        else:
            x = obj.x
            mask = getattr(obj, 'mask', None)
            obj.y = x
            obj = {"x": x, "mask": mask, "y": x}

        x = self.feature_grouping(obj, mask=mask)
        x = self.group_propagation(x)
        obj["x"] = x
        x = self.feature_ungrouping(obj, mask=mask)

        return x
