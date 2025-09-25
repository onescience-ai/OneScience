import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from onescience.datapipes.eagle import EagleDataset, collate
from onescience.distributed.manager import DistributedManager
from onescience.models.graphvit import GraphViT

# 参数解析器设置
parser = argparse.ArgumentParser(description="Train GraphViT model on Eagle dataset")

# 数据集相关参数
parser.add_argument(
    "--dataset-path",
    default="./Eagle_dataset/",
    type=Path,
    help="Path to the Eagle dataset directory",
)
parser.add_argument(
    "--cluster-path",
    default="./Eagle_dataset/",
    type=Path,
    help="Path to cluster data directory",
)
parser.add_argument(
    "--splits-path", default="./splits/", type=Path, help="Path to dataset split files"
)
parser.add_argument(
    "--model-name",
    type=str,
    required=True,
    help="Name for the saved model checkpoint (without extension)",
)
parser.add_argument(
    "--output-path", type=Path, required=True, help="Path to save model checkpoints"
)

# 模型与训练参数
parser.add_argument(
    "--n-cluster",
    type=int,
    required=True,
    help="Number of clusters for graph partitioning",
)
parser.add_argument("--epoch", default=1000, type=int, help="Number of training epochs")
parser.add_argument(
    "--lr", default=1e-4, type=float, help="Learning rate for optimizer"
)
parser.add_argument(
    "--horizon-val",
    default=25,
    type=int,
    help="Time window length for validation sequences",
)
parser.add_argument(
    "--horizon-train",
    default=6,
    type=int,
    help="Time window length for training sequences",
)
parser.add_argument(
    "--w-size",
    default=512,
    type=int,
    help="Window size/embedding dimension for GraphViT model",
)
parser.add_argument(
    "--alpha",
    default=0.1,
    type=float,
    help="Weighting factor for pressure loss component",
)
parser.add_argument("--batch-size", default=2, type=int, help="Batch size for training")

args = parser.parse_args()


def get_loss(velocity, pressure, output, state_hat, target, mask):
    velocity = velocity[:, 1:]
    pressure = pressure[:, 1:]
    velocity_hat = state_hat[:, 1:, :, :2]
    mask = mask[:, 1:].unsqueeze(-1)

    rmse_velocity = torch.sqrt(
        ((velocity * mask - velocity_hat * mask) ** 2).mean(dim=(-1))
    )
    loss_velocity = torch.mean(rmse_velocity)
    losses = {}

    pressure_hat = state_hat[:, 1:, :, 2:]
    rmse_pressure = torch.sqrt(
        ((pressure * mask - pressure_hat * mask) ** 2).mean(dim=(-1))
    )
    loss_pressure = torch.mean(rmse_pressure)
    MSE = nn.MSELoss()
    loss = MSE(target[..., :2] * mask, output[..., :2] * mask) + args.alpha * MSE(
        target[..., 2:] * mask, output[..., 2:] * mask
    )
    loss = loss

    losses["MSE_pressure"] = loss_pressure
    losses["loss"] = loss
    losses["MSE_velocity"] = loss_velocity

    return losses


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    epoch: int = 0,
    device: torch.device = None,
):
    with torch.no_grad():
        total_loss, cpt = 0, 0
        model.eval()
        for i, x in enumerate(tqdm(dataloader, desc="Validation")):
            mesh_pos = x["mesh_pos"].to(device)
            edges = x["edges"].to(device).long()
            velocity = x["velocity"].to(device)
            pressure = x["pressure"].to(device)
            node_type = x["node_type"].to(device)
            mask = x["mask"].to(device)
            clusters = x["cluster"].to(device).long()
            clusters_mask = x["cluster_mask"].to(device).long()

            state = torch.cat([velocity, pressure], dim=-1)
            state_hat, output, target = model(
                mesh_pos,
                edges,
                state,
                node_type,
                clusters,
                clusters_mask,
                apply_noise=False,
            )

            state_hat[..., :2], state_hat[..., 2:] = dataloader.dataset.denormalize(
                state_hat[..., :2], state_hat[..., 2:]
            )
            velocity, pressure = dataloader.dataset.denormalize(velocity, pressure)

            costs = get_loss(velocity, pressure, output, state_hat, target, mask)
            total_loss += costs["loss"].item()
            cpt += mesh_pos.shape[0]
    results = total_loss / cpt
    print(f"=== EPOCH {epoch + 1} ===\n{results}")
    return results


def main():
    DistributedManager.initialize()
    dist = DistributedManager()
    if dist.rank == 0:
        print(
            f"Initialized process group: rank {dist.rank}, world size {dist.world_size}"
        )
        print(args)
    device = dist.device
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    train_dataset = EagleDataset(
        args.dataset_path,
        cluster_path=args.cluster_path,
        splits_path=args.splits_path,
        split="train",
        window_length=args.horizon_train,
        n_cluster=args.n_cluster,
        normalized=True,
    )
    valid_dataset = EagleDataset(
        args.dataset_path,
        cluster_path=args.cluster_path,
        splits_path=args.splits_path,
        split="valid",
        window_length=args.horizon_val,
        n_cluster=args.n_cluster,
        normalized=True,
    )

    # 创建分布式采样器
    train_sampler = DistributedSampler(train_dataset) if dist.world_size > 1 else None
    valid_sampler = (
        DistributedSampler(valid_dataset, shuffle=False)
        if dist.world_size > 1
        else None
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=1,
        pin_memory=False,
        collate_fn=collate,
        sampler=train_sampler,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        collate_fn=collate,
        sampler=valid_sampler,
    )

    model = GraphViT(state_size=4, w_size=args.w_size).to(device)
    if dist.world_size > 1:
        model = DDP(model, device_ids=[dist.device])

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    if dist.rank == 0:
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("#params:", params)

    memory = torch.inf
    output_ckpt = args.output_path / args.model_name
    output_ckpt = output_ckpt.with_suffix(".nn")
    if dist.rank == 0:
        output_ckpt.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epoch):
        if dist.world_size > 1:
            train_sampler.set_epoch(epoch)
        model.train()

        for i, x in enumerate(
            tqdm(train_dataloader, desc="Training", disable=dist.rank != 0)
        ):
            mesh_pos = x["mesh_pos"].to(device)
            edges = x["edges"].to(device).long()
            velocity = x["velocity"].to(device)
            pressure = x["pressure"].to(device)
            node_type = x["node_type"].to(device)
            mask = x["mask"].to(device)
            clusters = x["cluster"].to(device).long()
            clusters_mask = x["cluster_mask"].to(device).long()

            state = torch.cat([velocity, pressure], dim=-1)
            state_hat, output, target = model(
                mesh_pos,
                edges,
                state,
                node_type,
                clusters,
                clusters_mask,
                apply_noise=True,
            )

            state_hat[..., :2], state_hat[..., 2:] = train_dataset.denormalize(
                state_hat[..., :2], state_hat[..., 2:]
            )
            velocity, pressure = train_dataset.denormalize(velocity, pressure)

            costs = get_loss(velocity, pressure, output, state_hat, target, mask)

            optim.zero_grad()
            costs["loss"].backward()
            optim.step()
        if dist.rank == 0:
            error = validate(model, valid_dataloader, epoch=epoch, device=dist.device)
            if error < memory:
                memory = error
                model_to_save = model.module if hasattr(model, "module") else model
                torch.save(model_to_save.state_dict(), output_ckpt)
                print("Saved!")
        else:
            error = float("inf")

        # 同步所有进程
        if dist.world_size > 1:
            torch.distributed.barrier()
    if dist.rank == 0:
        validate(model, valid_dataloader, device=dist.device)
    # 清理分布式进程
    dist.cleanup()


if __name__ == "__main__":
    main()
