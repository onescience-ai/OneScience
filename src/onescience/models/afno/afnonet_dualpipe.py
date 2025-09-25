import torch
import torch.nn as nn
from einops import rearrange

import onescience.distributed.comm as comm
from onescience.distributed.dualpipev import DualPipeV
from onescience.models.afno.afnonet_ocean import AFNONet


def set_p2p_tensor_shapes(tensors):
    comm.TENSOR_SHAPES = [
        tuple(t.shape)
        for t in (tensors if isinstance(tensors, (list, tuple)) else [tensors])
    ]


def set_p2p_tensor_dtype(tensor):
    comm.TENSOR_DTYPE = tensor.dtype


def extract_stage_modules(model, block_partition, stage_index):
    class PatchStage(nn.Module):
        def __init__(self, patch_embed, pos_embed, pos_drop, img_h, img_w, embed_dim):
            super().__init__()
            self.patch_embed = patch_embed
            self.pos_drop = pos_drop
            self.pos_embed = pos_embed
            self.h = img_h
            self.w = img_w
            self.embed_dim = embed_dim

        def forward(self, x):
            B = x.shape[0]
            x = self.patch_embed(x)
            x = x + self.pos_embed
            x = self.pos_drop(x)
            # x = x.reshape(B, self.h, self.w, self.embed_dim).contiguous()
            # return x
            out = x.reshape(B, self.h, self.w, self.embed_dim).clone()
            return out

    class HeadWrap(nn.Module):
        def __init__(self, head, img_size, patch_size, out_chans):
            super().__init__()
            self.head = head
            self.img_size = img_size
            self.patch_size = patch_size
            self.out_chans = out_chans

        def forward(self, x):
            x = self.head(x)
            x = rearrange(
                x,
                "b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
                p1=self.patch_size[0],
                p2=self.patch_size[1],
                h=self.img_size[0] // self.patch_size[0],
                w=self.img_size[1] // self.patch_size[1],
            )
            return x.contiguous()

    total_blocks = len(model.blocks)
    assert sum(block_partition) == total_blocks

    all_stages = []
    all_stages.append(
        PatchStage(
            model.patch_embed,
            model.pos_embed,
            model.pos_drop,
            model.h,
            model.w,
            model.embed_dim,
        )
    )

    start = 0
    for count in block_partition:
        end = start + count
        all_stages.append(nn.Sequential(*model.blocks[start:end]))
        start = end

    all_stages.append(
        HeadWrap(model.head, model.img_size, model.patch_size, model.out_chans)
    )
    return all_stages[stage_index]


def build_dualpipe_model(params, rank, world_size, existing_model=None):
    torch.cuda.set_device(rank)
    backend = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(f"{backend}:{rank}")
    # print(f"[Rank {rank}] Using device: {device}")

    model = existing_model if existing_model is not None else AFNONet(params)
    block_partition = getattr(params, "block_partition", None)
    assert block_partition is not None, "Missing block_partition in params"

    total_stages = len(block_partition) + 2
    assert (
        total_stages == world_size * 2
    ), "DualPipe expects total_stages = world_size * 2"

    stage_map = list(range(world_size)) + list(reversed(range(world_size)))
    assigned_stages = [i for i, r in enumerate(stage_map) if r == rank]
    assert len(assigned_stages) == 2, "Each rank must be assigned exactly two stages"

    phase0 = extract_stage_modules(model, block_partition, assigned_stages[0]).to(
        device
    )
    phase1 = extract_stage_modules(model, block_partition, assigned_stages[1]).to(
        device
    )

    module = DualPipeV((phase0, phase1))

    return module
