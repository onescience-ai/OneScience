import math
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass

from onescience.models.meta import ModelMetaData
from onescience.modules import OneEmbedding, OneRecovery, OneSample
from onescience.modules.fuser.onefuser import OneFuser
from onescience.modules.module import Module

from onescience.distributed.megatron.training import get_args
from onescience.distributed.megatron.training.arguments import core_transformer_config_from_args
from onescience.distributed.megatron.core.tensor_parallel import checkpoint


@dataclass
class MetaData(ModelMetaData):
    name: str = "Xihe"
    jit: bool = False
    cuda_graphs: bool = True
    amp: bool = True
    onnx_cpu: bool = False
    onnx_gpu: bool = True
    onnx_runtime: bool = True
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False


class TensorWithMask:
    def __init__(self, x, mask):
        self.x = x
        self.mask = mask
        self.y = None


def change_mask(mask_full, x, h_out, w_out):
    if not torch.is_tensor(mask_full):
        if isinstance(mask_full, str):
            mask_full = np.load(mask_full)
        mask_full = torch.tensor(mask_full, dtype=torch.float32)

    H, W = mask_full.shape
    patch_h = math.ceil(H / h_out)
    patch_w = math.ceil(W / w_out)

    mask_coarse = torch.zeros((h_out, w_out), dtype=torch.float32)
    for i in range(h_out):
        for j in range(w_out):
            h0, h1 = i * patch_h, min((i + 1) * patch_h, H)
            w0, w1 = j * patch_w, min((j + 1) * patch_w, W)
            patch = mask_full[h0:h1, w0:w1]
            mask_coarse[i, j] = 1.0 if torch.any(patch > 0.5) else 0.0

    mask_coarse = mask_coarse.to(x.device, dtype=x.dtype)
    B = x.shape[0]
    mask_coarse = mask_coarse.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)
    return mask_coarse


class Xihe_stage0(Module):
    def __init__(
        self,
        img_size=(2041, 4320),
        patch_size=(6, 12),
        window_size=(6, 12),
        embed_dim=192,
        num_heads=(6, 12, 12, 6),
        in_chans=96,
        depth=2,
        mask_full=None,
        out_chans=96,
        num_groups=64,
        config=None,
        **kwargs,
    ):
        super().__init__(meta=MetaData())
        self.pre_process = None
        self.share_embeddings_and_output_weights = None
        self.input_tensor = None

        args = get_args()
        self.config = core_transformer_config_from_args(args) if config is None else config

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.mask_full = mask_full
        if isinstance(mask_full, str):
            self.mask_full = np.load(mask_full)

        self.patchembed2d = OneEmbedding(style="XiheEmbedding")

        H_out = math.ceil(img_size[0] / patch_size[0])
        W_out = math.ceil(img_size[1] / patch_size[1])
        input_resolution = (1, H_out, W_out)

        window_size_3d = (1, window_size[0], window_size[1])
        num_heads_local = num_heads[0]

        self.block1 = OneFuser(
            dim=embed_dim,
            input_resolution=input_resolution,
            num_local=1,
            num_global=0,
            num_heads_local=num_heads_local,
            window_size=window_size_3d,
            style="XiheDistributedFuser",
            config=self.config,
        )

        self.downsample = OneSample(
            style="PanguDownSample2D",
            in_dim=embed_dim,
            input_resolution=(H_out, W_out),
            output_resolution=(H_out // 2, W_out // 2),
        )

        input_resolution_half = (1, H_out // 2, W_out // 2)
        num_heads_local_half = num_heads[1]

        self.block2 = OneFuser(
            dim=2 * embed_dim,
            input_resolution=input_resolution_half,
            num_local=2,
            num_global=1,
            num_heads_local=num_heads_local_half,
            num_heads_global=num_heads[1],
            window_size=window_size_3d,
            style="XiheDistributedFuser",
            num_groups=num_groups,
            config=self.config,
        )

        self.block3 = OneFuser(
            dim=2 * embed_dim,
            input_resolution=input_resolution_half,
            num_local=2,
            num_global=1,
            num_heads_local=num_heads_local_half,
            num_heads_global=num_heads[1],
            window_size=window_size_3d,
            style="XiheDistributedFuser",
            num_groups=num_groups,
            config=self.config,
        )

    def set_input_tensor(self, input_tensor):
        self.input_tensor = input_tensor

    def forward(self, x):
        x = self.patchembed2d(x)
        x = x.flatten(2).transpose(1, 2)
        B, N, C = x.shape

        mask_full = self.mask_full
        H_out = math.ceil(self.img_size[0] / self.patch_size[0])
        W_out = math.ceil(self.img_size[1] / self.patch_size[1])

        if mask_full is not None:
            mask1 = change_mask(mask_full, x, h_out=H_out, w_out=W_out)
        else:
            mask1 = None

        obj1 = TensorWithMask(x, mask1)
        x = self.block1(obj1)
        x1 = x

        x = self.downsample(x)

        H_out_half = H_out // 2
        W_out_half = W_out // 2
        if mask_full is not None:
            mask2 = change_mask(mask_full, x, h_out=H_out_half, w_out=W_out_half)
        else:
            mask2 = None

        obj2 = TensorWithMask(x, mask2)
        x = self.block2(obj2)
        obj2 = TensorWithMask(x, mask2)
        x = self.block3(obj2)

        return (x, x1)


class Xihe_stage1(Module):
    def __init__(
        self,
        img_size=(2041, 4320),
        patch_size=(6, 12),
        window_size=(6, 12),
        embed_dim=192,
        num_heads=(6, 12, 12, 6),
        in_chans=96,
        depth=2,
        mask_full=None,
        out_chans=96,
        num_groups=64,
        config=None,
        **kwargs,
    ):
        super().__init__(meta=MetaData())
        self.pre_process = None
        self.share_embeddings_and_output_weights = None
        self.input_tensor = None

        args = get_args()
        self.config = core_transformer_config_from_args(args) if config is None else config

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.out_chans = out_chans
        self.mask_full = mask_full
        if isinstance(mask_full, str):
            self.mask_full = np.load(mask_full)

        H_out = math.ceil(img_size[0] / patch_size[0])
        W_out = math.ceil(img_size[1] / patch_size[1])
        input_resolution_half = (1, H_out // 2, W_out // 2)
        window_size_3d = (1, window_size[0], window_size[1])

        num_heads_local_half = num_heads[2]

        self.block4 = OneFuser(
            dim=2 * embed_dim,
            input_resolution=input_resolution_half,
            num_local=2,
            num_global=1,
            num_heads_local=num_heads_local_half,
            num_heads_global=num_heads[2],
            window_size=window_size_3d,
            style="XiheDistributedFuser",
            num_groups=num_groups,
            config=self.config,
        )

        self.upsample = OneSample(
            style="XiheUpSample",
            in_dim=2 * embed_dim,
            out_dim=embed_dim,
            input_resolution=(H_out // 2, W_out // 2),
            output_resolution=(H_out, W_out),
        )

        input_resolution_full = (1, H_out, W_out)
        num_heads_local_full = num_heads[3]

        self.block5 = OneFuser(
            dim=embed_dim,
            input_resolution=input_resolution_full,
            num_local=1,
            num_global=0,
            num_heads_local=num_heads_local_full,
            window_size=window_size_3d,
            style="XiheDistributedFuser",
            config=self.config,
        )

        self.skip_proj = nn.Linear(2 * embed_dim, embed_dim)

        self.patchrecovery2d = OneRecovery(style="XihePatchRecovery")

    def set_input_tensor(self, input_tensor):
        self.input_tensor = input_tensor

    def forward(self, x):
        if self.input_tensor is not None:
            if isinstance(self.input_tensor, list):
                x, x1 = tuple(self.input_tensor)
            else:
                x, x1 = self.input_tensor

        mask_full = self.mask_full
        H_out = math.ceil(self.img_size[0] / self.patch_size[0])
        W_out = math.ceil(self.img_size[1] / self.patch_size[1])

        H_out_half = H_out // 2
        W_out_half = W_out // 2

        if mask_full is not None:
            mask2 = change_mask(mask_full, x, h_out=H_out_half, w_out=W_out_half)
        else:
            mask2 = None

        obj2 = TensorWithMask(x, mask2)
        x = self.block4(obj2)

        x = self.upsample(x)

        if mask_full is not None:
            mask1 = change_mask(mask_full, x, h_out=H_out, w_out=W_out)
        else:
            mask1 = None

        obj1 = TensorWithMask(x, mask1)
        x = self.block5(obj1)

        B, N, C = x.shape
        x_out = torch.cat([x, x1], dim=-1)
        x_out = self.skip_proj(x_out)
        H_ = math.ceil(self.img_size[0] / self.patch_size[0])
        W_ = math.ceil(self.img_size[1] / self.patch_size[1])
        x_out = x_out.transpose(1, 2).reshape(B, C, H_, W_)
        x = self.patchrecovery2d(x_out)
        return x
