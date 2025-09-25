import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from args import Args
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from onescience.distributed.manager import DistributedManager
from onescience.models.cfdbench.base_model import CfdModel
from onescience.models.cfdbench.deeponet import DeepONet
from onescience.models.cfdbench.ffn import FfnModel
from onescience.models.cfdbench.loss import loss_name_to_fn
from onescience.utils.cfdbench.dataset import CfdDataset, get_dataset
from onescience.utils.cfdbench.utils import (
    dump_json,
    get_output_dir,
    load_best_ckpt,
    plot_loss,
    plot_predictions,
)


def collate_fn(batch: list):
    case_params, t, label = zip(*batch)
    case_params = torch.stack(case_params)  # (b, p)
    label = torch.stack(label)  # (b, c, h, w), c=2
    t = torch.stack(t)  # (b, 1)
    return dict(
        case_params=case_params.cuda(),
        t=t.cuda(),
        label=label.cuda(),
    )


def evaluate(
    model: CfdModel,
    data: CfdDataset,
    output_dir: Path,
    batch_size: int = 64,
    plot_interval: int = 1,
    measure_time: bool = False,
) -> Dict[str, Any]:
    # Only rank 0 process performs evaluation
    dist = DistributedManager()
    if dist.world_size > 1 and dist.rank != 0:
        return {"scores": {}, "preds": []}

    # Unwrap DDP model if necessary
    model_eval = model.module if hasattr(model, "module") else model
    model_eval.eval()

    if measure_time:
        assert batch_size == 1

    loader = DataLoader(
        data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    scores = {name: [] for name in model_eval.loss_fn.get_score_names()}
    all_preds = []

    print(f"=== Evaluating (rank {dist.rank}) ===")
    print(f"# examples: {len(data)}")
    print(f"batch size: {batch_size}")
    print(f"# batches: {len(loader)}")
    print(f"Plot interval: {plot_interval}")
    start_time = time.time()
    with torch.no_grad():
        for step, batch in enumerate(tqdm(loader)):
            case_params = batch["case_params"]  # (b, 5)
            label = batch["label"]
            t = batch["t"]  # (b, 1)

            height, width = label.shape[-2:]

            # Compute the prediction and its loss
            preds = model_eval.generate_one(
                case_params=case_params, t=t, height=height, width=width
            )
            loss: dict = model_eval.loss_fn(labels=label[:, :1], preds=preds)
            for key in scores:
                scores[key].append(loss[key].item())

            preds = preds.repeat(1, 3, 1, 1)
            all_preds.append(preds.cpu().detach())
            if step % plot_interval == 0 and not measure_time:
                # Dump input, label and prediction flow images.
                image_dir = output_dir / "images"
                image_dir.mkdir(exist_ok=True, parents=True)
                plot_predictions(
                    inp=None,
                    label=label[0][0],
                    pred=preds[0][0],
                    out_dir=image_dir,
                    step=step,
                )

    if measure_time:
        print("Memory usage:")
        print(torch.cuda.memory_summary("cuda"))
        print("Time usage:")
        time_per_step = 1000 * (time.time() - start_time) / len(loader)
        print(f"Time per step: {time_per_step:.3f} ms")
        exit()

    avg_scores = {key: np.mean(vals) for key, vals in scores.items()}
    for key, vals in scores.items():
        print(f"{key}: {np.mean(vals)}")

    plot_loss(scores["nmse"], output_dir / "loss.png")
    return dict(
        scores=dict(
            mean=avg_scores,
            all=scores,
        ),
        preds=all_preds,
    )


def test(
    model: CfdModel,
    data: CfdDataset,
    output_dir: Path,
    plot_interval: int = 10,
    batch_size: int = 1,
    measure_time: bool = False,
):
    """
    Perform inference on the test set.
    """
    # Only rank 0 process performs testing
    dist = DistributedManager()
    if dist.world_size > 1 and dist.rank != 0:
        return

    assert plot_interval > 0
    if dist.rank == 0:
        output_dir.mkdir(exist_ok=True, parents=True)
    print(f"==== Testing (rank {dist.rank}) ====")
    print(f"Batch size: {batch_size}")
    print(f"Plot interval: {plot_interval}")
    result = evaluate(
        model,
        data,
        output_dir,
        batch_size=batch_size,
        plot_interval=plot_interval,
        measure_time=measure_time,
    )
    preds = result["preds"]
    scores = result["scores"]
    if dist.rank == 0:
        torch.save(preds, output_dir / "preds.pt")
        dump_json(scores, output_dir / "scores.json")
    print("==== Testing done ====")


def train(
    model: CfdModel,
    train_data: CfdDataset,
    dev_data: CfdDataset,
    output_dir: Path,
    num_epochs: int = 400,
    lr: float = 1e-3,
    lr_step_size: int = 1,
    lr_gamma: float = 0.9,
    batch_size: int = 64,
    log_interval: int = 50,
    eval_interval: int = 2,
    measure_time: bool = False,
):
    dist = DistributedManager()

    # Create distributed sampler and data loader
    sampler = DistributedSampler(train_data) if dist.world_size > 1 else None
    loader = DataLoader(
        train_data,
        batch_size=batch_size,
        collate_fn=collate_fn,
        sampler=sampler,
        shuffle=(sampler is None),
    )

    if dist.rank == 0:
        output_dir.mkdir(exist_ok=True, parents=True)
        print("==== Training ====")
        print(f"Output dir: {output_dir}")
        print(f"# lr: {lr}")
        print(f"# batch: {batch_size} (per GPU)")
        print(f"# examples: {len(train_data)}")
        print(f"# step: {len(loader)}")
        print(f"# epoch: {num_epochs}")
        print(f"# GPUs: {dist.world_size}")

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    start_time = time.time()
    global_step = 0
    all_train_losses = []

    for ep in range(num_epochs):
        if sampler is not None:
            sampler.set_epoch(ep)
        ep_start_time = time.time()
        ep_train_losses = []
        model.train()

        for step, batch in enumerate(loader):
            # Forward
            outputs = model(**batch)
            losses = outputs["loss"]
            loss = losses["nmse"]

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log
            ep_train_losses.append(loss.item())
            global_step += 1

            # Only log from rank 0
            if dist.rank == 0 and global_step % log_interval == 0 and not measure_time:
                avg_loss = np.mean(ep_train_losses[-log_interval:])
                log_info = {
                    "ep": ep,
                    "step": step,
                    "loss": f"{avg_loss:.3e}",
                    "lr": f"{scheduler.get_last_lr()[0]:.3e}",
                    "time": int(time.time() - start_time),
                }
                print(log_info)

        if measure_time:
            if dist.rank == 0:
                print("Memory usage:")
                print(torch.cuda.memory_summary("cuda"))
                print("Time usage:")
                print(time.time() - ep_start_time)
            exit()

        scheduler.step()

        # Evaluate on rank 0 only
        if dist.rank == 0 and (ep + 1) % eval_interval == 0:
            ckpt_dir = output_dir / f"ckpt-{ep}"
            ckpt_dir.mkdir(exist_ok=True, parents=True)
            dev_result = evaluate(model, dev_data, ckpt_dir)
            dev_scores = dev_result["scores"]
            dump_json(dev_scores, ckpt_dir / "dev_loss.json")
            dump_json(ep_train_losses, ckpt_dir / "train_loss.json")

            # Save checkpoint - unwrap DDP model
            model_to_save = model.module if hasattr(model, "module") else model
            ckpt_path = ckpt_dir / "model.pt"
            print(f"Saving checkpoint to {ckpt_path}")
            torch.save(model_to_save.state_dict(), ckpt_path)

            # Save average scores
            ep_scores = dict(
                ep=ep,
                train_loss=np.mean(ep_train_losses),
                dev_loss=np.mean(dev_scores["mean"]["nmse"]),
                time=time.time() - ep_start_time,
            )
            dump_json(ep_scores, ckpt_dir / "scores.json")

        # All processes wait for evaluation to finish
        if dist.world_size > 1:
            torch.distributed.barrier()

        all_train_losses.append(ep_train_losses)

    # Only save from rank 0
    if dist.rank == 0:
        all_train_losses = sum(all_train_losses, [])
        dump_json(all_train_losses, output_dir / "train_losses.json")
        plot_loss(all_train_losses, output_dir / "train_losses.png")


def init_model(args: Args) -> CfdModel:
    """
    Instantiate a nonautoregressive model.
    """
    loss_fn = loss_name_to_fn(args.loss_name)
    query_coord_dim = 3  # (t, x, y)
    if "cylinder" in args.data_name:
        # (density, viscosity, u_top, h, w, radius, center_x, center_y)
        n_case_params = 8
    else:
        n_case_params = 5  # (density, viscosity, u_top, h, w)
    if args.model == "deeponet":
        model = DeepONet(
            branch_dim=n_case_params,
            trunk_dim=query_coord_dim,
            loss_fn=loss_fn,
            width=args.deeponet_width,
            trunk_depth=args.trunk_depth,
            branch_depth=args.branch_depth,
            act_name=args.act_fn,
            act_norm=bool(args.act_scale_invariant),
            act_on_output=bool(args.act_on_output),
        )
    elif args.model == "ffn":
        widths = (
            [n_case_params + query_coord_dim] + [args.ffn_width] * args.ffn_depth + [1]
        )
        model = FfnModel(
            widths=widths,
            loss_fn=loss_fn,
        )
    else:
        raise ValueError(f"Invalid model name: {args.model}")
    sum(p.numel() for p in model.parameters())
    return model


def main():
    # Initialize distributed environment
    DistributedManager.initialize()
    dist = DistributedManager()
    print(f"Initialized process group: rank {dist.rank}, world size {dist.world_size}")

    args = Args().parse_args()
    if dist.rank == 0:
        print(args)

    output_dir = get_output_dir(args)
    if dist.rank == 0:
        output_dir.mkdir(exist_ok=True, parents=True)
        args.save(str(output_dir / "args.json"))

    # Data
    if dist.rank == 0:
        print("Loading data...")
    data_dir = Path(args.data_dir)
    train_data, dev_data, test_data = get_dataset(
        data_dir=data_dir,
        data_name=args.data_name,
        norm_props=bool(args.norm_props),
        norm_bc=bool(args.norm_bc),
        rank=dist.rank,
    )
    if dist.rank == 0:
        print(f"# train examples: {len(train_data)}")
        print(f"# dev examples: {len(dev_data)}")
        print(f"# test examples: {len(test_data)}")

    # Model
    if dist.rank == 0:
        print("Loading model")
    model = init_model(args)
    model = model.to(dist.device)

    # Wrap model with DDP for training
    if "train" in args.mode and dist.world_size > 1:
        model = DDP(model, device_ids=[dist.device])

    # Training
    if "train" in args.mode:
        if dist.rank == 0:
            args.save(str(output_dir / "train_args.json"))
        train(
            model,
            train_data,
            dev_data,
            output_dir,
            batch_size=args.batch_size,
            lr=args.lr,
            lr_step_size=args.lr_step_size,
            num_epochs=args.num_epochs,
            eval_interval=args.eval_interval,
        )

    # Testing
    if "test" in args.mode and dist.rank == 0:
        args.save(str(output_dir / "test_args.json"))
        # Test - if model is DDP, we need to unwrap it first
        model_to_test = model.module if hasattr(model, "module") else model
        load_best_ckpt(model_to_test, output_dir)
        test_dir = output_dir / "test"
        test_dir.mkdir(exist_ok=True)
        test(
            model_to_test,
            data=test_data,
            output_dir=test_dir,
            batch_size=1,
            plot_interval=10,
        )


if __name__ == "__main__":
    main()
