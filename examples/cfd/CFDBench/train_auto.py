import time
from copy import deepcopy
from pathlib import Path
from shutil import copyfile
from typing import List

import numpy as np
import torch
from args import Args
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from onescience.distributed.manager import DistributedManager
from onescience.models.cfdbench.base_model import AutoCfdModel
from onescience.utils.cfdbench.dataset import get_auto_dataset
from onescience.utils.cfdbench.dataset.base import CfdAutoDataset
from onescience.utils.cfdbench.utils import (
    dump_json,
    get_output_dir,
    load_best_ckpt,
    plot_loss,
    plot_predictions,
)
from onescience.utils.cfdbench.utils_auto import init_model


def collate_fn(batch: list):
    # batch is a list of tuples (input_frame, label_frame, case_params)
    inputs, labels, case_params = zip(*batch)
    inputs = torch.stack(inputs)  # (b, 3, h, w)
    labels = torch.stack(labels)  # (b, 3, h, w)

    # The last channel from features is the binary mask.
    labels = labels[:, :-1]  # (b, 2, h, w)
    mask = inputs[:, -1:]  # (b, 1, h, w)
    inputs = inputs[:, :-1]  # (b, 2, h, w)

    # Case params is a dict, turn it into a tensor
    keys = [x for x in case_params[0].keys() if x not in ["rotated", "dx", "dy"]]
    case_params_vec = []
    for case_param in case_params:
        case_params_vec.append([case_param[k] for k in keys])
    case_params = torch.tensor(case_params_vec)  # (b, 5)
    # Build the kwargs dict for the model's forward method
    return dict(
        inputs=inputs.cuda(),
        label=labels.cuda(),
        mask=mask.cuda(),
        case_params=case_params.cuda(),
    )


def evaluate(
    model: AutoCfdModel,
    data: CfdAutoDataset,
    output_dir: Path,
    batch_size: int = 2,
    plot_interval: int = 1,
    measure_time: bool = False,
):
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
    input_scores = deepcopy(scores)
    all_preds: List[Tensor] = []

    print(f"=== Evaluating (rank {dist.rank}) ===")
    print(f"# examples: {len(data)}")
    print(f"Batch size: {batch_size}")
    print(f"# batches: {len(loader)}")
    print(f"Plot interval: {plot_interval}")
    print(f"Output dir: {output_dir}")

    start_time = time.time()
    with torch.inference_mode():
        for step, batch in enumerate(tqdm(loader)):
            # inputs, labels, case_params = batch
            inputs = batch["inputs"]  # (b, 2, h, w)
            labels = batch["label"]  # (b, 2, h, w)

            # Compute difference between the input and label
            input_loss: dict = model_eval.loss_fn(
                labels=labels[:, :1], preds=inputs[:, :1]
            )
            for key in input_scores:
                input_scores[key].append(input_loss[key].cpu().tolist())

            # Compute the prediction and its loss
            outputs: dict = model_eval(**batch)
            loss: dict = outputs["loss"]
            preds: Tensor = outputs["preds"]
            height, width = labels.shape[2:]

            # When using DeepONetAuto, the prediction is a flattened.
            preds = preds.view(-1, 1, height, width)  # (b, 1, h, w)
            for key in scores:
                scores[key].append(loss[key].cpu().tolist())

            all_preds.append(preds.cpu().detach())

            if step % plot_interval == 0 and not measure_time:
                # Dump input, label and prediction flow images.
                image_dir = output_dir / "images"
                image_dir.mkdir(exist_ok=True, parents=True)
                plot_predictions(
                    inp=inputs[0][0],
                    label=labels[0][0],
                    pred=preds[0][0],
                    out_dir=image_dir,
                    step=step,
                )

    if measure_time:
        print("Memory usage:")
        print(torch.cuda.memory_summary("cuda"))
        print("Time usage:")
        time_per_step = 1000 * (time.time() - start_time) / len(loader)
        print(f"Time (ms) per step: {time_per_step:.3f}")
        exit()

    avg_scores = {}
    for key in scores:
        mean = np.mean(scores[key])
        input_mean = np.mean(input_scores[key])
        avg_scores[key] = mean
        avg_scores[f"input_{key}"] = input_mean
        print(f"Prediction {key}: {mean}")
        print(f"     Input {key}: {input_mean}")

    plot_loss(scores["nmse"], output_dir / "loss.png")
    return dict(
        preds=torch.cat(all_preds, dim=0),
        scores=dict(
            mean=avg_scores,
            all=scores,
        ),
    )


def test(
    model: AutoCfdModel,
    data: CfdAutoDataset,
    output_dir: Path,
    infer_steps: int = 200,
    plot_interval: int = 10,
    batch_size: int = 1,
    measure_time: bool = False,
):
    # Only rank 0 process performs testing
    dist = DistributedManager()
    if dist.world_size > 1 and dist.rank != 0:
        return

    assert infer_steps > 0
    assert plot_interval > 0
    if dist.rank == 0:
        output_dir.mkdir(exist_ok=True, parents=True)

    print(f"=== Testing (rank {dist.rank}) ===")
    print(f"batch_size: {batch_size}")
    print(f"Plot interval: {plot_interval}")

    result = evaluate(
        model,
        data,
        output_dir=output_dir,
        batch_size=batch_size,
        plot_interval=plot_interval,
        measure_time=measure_time,
    )

    preds = result["preds"]
    scores = result["scores"]

    if dist.rank == 0:
        torch.save(preds, output_dir / "preds.pt")
        dump_json(scores, output_dir / "scores.json")

    print("=== Testing done ===")


def train(
    model: AutoCfdModel,
    train_data: CfdAutoDataset,
    dev_data: CfdAutoDataset,
    output_dir: Path,
    num_epochs: int = 400,
    lr: float = 1e-3,
    lr_step_size: int = 1,
    lr_gamma: float = 0.9,
    batch_size: int = 2,
    eval_batch_size: int = 2,
    log_interval: int = 10,
    eval_interval: int = 2,
    measure_time: bool = False,
):
    dist = DistributedManager()

    # Create distributed sampler and data loader
    sampler = DistributedSampler(train_data) if dist.world_size > 1 else None
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        collate_fn=collate_fn,
        sampler=sampler,
        shuffle=False,  # 由sampler控制shuffle
        drop_last=True,  # 添加drop_last确保批次大小一致
    )
    if dist.rank == 0:
        output_dir.mkdir(exist_ok=True, parents=True)
        print("====== Training ======")
        print(f"# batch: {batch_size} (per GPU)")
        print(f"# examples: {len(train_data)}")
        print(f"# step: {len(train_loader)}")
        print(f"# epoch: {num_epochs}")
        print(f"# GPUs: {dist.world_size}")

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    start_time = time.time()
    global_step = 0
    train_losses = []

    for ep in range(num_epochs):
        # Set epoch for distributed sampler
        if sampler is not None:
            sampler.set_epoch(ep)

        ep_start_time = time.time()
        ep_train_losses = []
        model.train()

        for step, batch in enumerate(train_loader):
            # Forward
            outputs: dict = model(**batch)

            # Backward
            loss: dict = outputs["loss"]
            loss["nmse"].backward()
            optimizer.step()
            optimizer.zero_grad()

            # Log
            ep_train_losses.append(loss["nmse"].item())
            global_step += 1
            # Only log from rank 0
            if dist.rank == 0 and (global_step % log_interval == 0):
                log_info = dict(
                    ep=ep,
                    step=step,
                    mse=f"{loss['mse'].item():.3e}",
                    nmse=f"{loss['nmse'].item():.3e}",
                    lr=f"{scheduler.get_last_lr()[0]:.3e}",
                    time=round(time.time() - start_time),
                )
                print(log_info)

        if measure_time:
            if dist.rank == 0:
                print("Memory usage:")
                print(torch.cuda.memory_summary("cuda"))
                print("Time usage:")
                print(time.time() - ep_start_time)
            exit()

        scheduler.step()
        train_losses += ep_train_losses

        # Plot and evaluate on rank 0 only
        if dist.rank == 0 and (ep + 1) % eval_interval == 0:
            ckpt_dir = output_dir / f"ckpt-{ep}"
            ckpt_dir.mkdir(exist_ok=True, parents=True)
            result = evaluate(model, dev_data, ckpt_dir, batch_size=eval_batch_size)
            dev_scores = result["scores"]
            dump_json(dev_scores, ckpt_dir / "dev_scores.json")
            dump_json(ep_train_losses, ckpt_dir / "train_loss.json")

            # Save checkpoint - unwrap DDP model
            model_to_save = model.module if hasattr(model, "module") else model
            ckpt_path = ckpt_dir / "model.pt"
            print(f"Saving checkpoint to {ckpt_path}")
            if ckpt_path.exists():
                ckpt_backup_path = ckpt_dir / "backup_model.pt"
                print(f"Backing up old checkpoint to {ckpt_backup_path}")
                copyfile(ckpt_path, ckpt_backup_path)
            torch.save(model_to_save.state_dict(), ckpt_path)

            # Save average scores
            ep_scores = dict(
                ep=ep,
                train_loss=np.mean(ep_train_losses),
                dev_loss=np.mean(dev_scores["all"]["nmse"]),  # type: ignore
                time=time.time() - ep_start_time,
            )
            dump_json(ep_scores, ckpt_dir / "scores.json")

        # All processes wait for evaluation to finish
        if dist.world_size > 1:
            torch.distributed.barrier()

    # Only rank 0 saves final training losses
    if dist.rank == 0:
        print("====== Training done ======")
        dump_json(train_losses, output_dir / "train_losses.json")
        plot_loss(train_losses, output_dir / "train_losses.png")


def main():
    # Initialize distributed environment
    DistributedManager.initialize()
    dist = DistributedManager()
    print(f"Initialized process group: rank {dist.rank}, world size {dist.world_size}")

    args = Args().parse_args()
    if dist.rank == 0:
        print("#" * 80)
        print(args)
        print("#" * 80)

    output_dir = get_output_dir(args, is_auto=True)
    if dist.rank == 0:
        output_dir.mkdir(exist_ok=True, parents=True)
        args.save(str(output_dir / "args.json"))

    # Data
    if dist.rank == 0:
        print("Loading data...")
    data_dir = Path(args.data_dir)
    train_data, dev_data, test_data = get_auto_dataset(
        data_dir=data_dir,
        data_name=args.data_name,
        delta_time=args.delta_time,
        norm_props=bool(args.norm_props),
        norm_bc=bool(args.norm_bc),
        rank=dist.rank,
    )
    assert train_data is not None
    assert dev_data is not None
    assert test_data is not None

    if dist.rank == 0:
        print(f"# train examples: {len(train_data)}")
        print(f"# dev examples: {len(dev_data)}")
        print(f"# test examples: {len(test_data)}")

    # Model
    if dist.rank == 0:
        print("Loading model")
    model = init_model(args)
    num_params = sum(p.numel() for p in model.parameters())
    if dist.rank == 0:
        print(f"Model has {num_params} parameters")

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
            train_data=train_data,
            dev_data=dev_data,
            output_dir=output_dir,
            lr=args.lr,
            lr_step_size=args.lr_step_size,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            eval_interval=args.eval_interval,
            log_interval=args.log_interval,
        )

    # Testing
    if "test" in args.mode and dist.rank == 0:
        args.save(str(output_dir / "test_args.json"))
        # Unwrap model if DDP
        model_to_test = model.module if hasattr(model, "module") else model
        load_best_ckpt(model_to_test, output_dir)
        test_dir = output_dir / "test"
        test_dir.mkdir(exist_ok=True)
        test(
            model_to_test,
            test_data,
            test_dir,
            batch_size=1,
            infer_steps=20,
            plot_interval=10,
        )


if __name__ == "__main__":
    main()
