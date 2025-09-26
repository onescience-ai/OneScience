# afnonet_pipeline.py
# This file is the pipepine-parallel version of afno, which is used as the baseline of pp-model performance tests.
import types

import torch
import torch.nn as nn
from einops import rearrange
from torch.distributed import init_process_group, rpc
from torch.distributed.pipeline.sync import Pipe

from onescience.models.afno.afnonet_ocean import AFNONet


def split_afnonet_into_stages(
    model: AFNONet, num_devices: int, devices: list, block_partition=None
):
    assert len(
        devices) == num_devices, "Each device must be uniquely assigned"
    assert (
        block_partition is not None
    ), "block_partition must be provided to control manual stage distribution"

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
            x = x.reshape(B, self.h, self.w, self.embed_dim)
            return x

    stages = []
    total_stages = len(block_partition) + 2
    total_stages // 2
    vstyle_device_order = list(
        range(num_devices)) + list(reversed(range(num_devices)))
    vstyle_device_order = vstyle_device_order[:total_stages]
    assert (
        len(vstyle_device_order) == total_stages
    ), "Device order should match total number of stages"

    stages.append(
        PatchStage(
            model.patch_embed,
            model.pos_embed,
            model.pos_drop,
            model.h,
            model.w,
            model.embed_dim,
        ).to(devices[vstyle_device_order[0]])
    )

    blocks = model.blocks
    total_blocks = len(blocks)
    assert (
        sum(block_partition) == total_blocks
    ), f"block_partition sum {sum(block_partition)} != total_blocks {total_blocks}"

    start = 0
    for i, count in enumerate(block_partition):
        end = start + count
        assigned_device = devices[vstyle_device_order[i + 1]]
        stage = nn.Sequential(
            *blocks[start:end]).to(assigned_device)
        stages.append(stage)
        start = end

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
            return x

    stages.append(
        HeadWrap(model.head, model.img_size, model.patch_size, model.out_chans).to(
            devices[vstyle_device_order[-1]]
        )
    )

    return nn.Sequential(*stages)


def build_pipeline_model(
    params,
    num_devices=4,
    chunks=4,
    batch_size=None,
    sample_input=None,
    existing_model=None,
):
    backend = "cuda" if torch.cuda.is_available() else "hip"
    devices = [torch.device(f"{backend}:{i}")
               for i in range(num_devices)]
    block_partition = getattr(
        params, "block_partition", None)
    depth = getattr(params, "depth", None)
    assert (
        depth is not None
    ), "Please include the {depth} in yaml file to identify the Blocks of afno"

    if sample_input is not None:
        inferred_batch_size = sample_input.shape[0]
        assert (
            inferred_batch_size >= chunks
        ), f"Inferred batch_size ({inferred_batch_size}) must be ≥ chunks ({chunks})"
    elif batch_size is not None:
        assert (
            batch_size >= chunks
        ), f"batch_size ({batch_size}) must be ≥ chunks ({chunks})"

    model = existing_model if existing_model is not None else AFNONet(
        params)

    staged_model = split_afnonet_into_stages(
        model, num_devices, devices, block_partition
    )
    return Pipe(staged_model, chunks=chunks, checkpoint="never")


# Initialize RPC for local testing (needed by Pipe even for single process)
if not rpc._is_current_rpc_agent_set():
    init_process_group(backend="gloo", rank=0, world_size=1)
    rpc.init_rpc("worker0", rank=0, world_size=1)

if __name__ == "__main__":

    def test_pipeline_partition_only():
        """
        ✅ used to test the pp-stage of afno
        ❌ not contain dualpipe, only contain pipelien
        """

        class DummyParams(types.SimpleNamespace):
            patch_size = 4
            image_width = 160
            image_height = 360
            in_channels = list(range(72))
            out_channels = list(range(48, 72))
            num_blocks = 8
            depth = 6
            in_chans = 72
            out_chans = 24
            mlp_ratio = 4.0
            drop_rate = 0.0
            drop_path_rate = 0.0
            block_partition = [1, 1, 1, 1, 1, 1]

        params = DummyParams()
        x = torch.randn(4, 72, 160, 360).to(
            "cuda:0")  # batch_size = 4
        model = build_pipeline_model(
            params, num_devices=4, chunks=4, sample_input=x)
        y = model(x).local_value()
        assert isinstance(
            y, torch.Tensor), "Output is not a tensor"
        print(
            "\u2705 Pipeline stage partition test successful. Output shape:", y.shape)

    test_pipeline_partition_only()
