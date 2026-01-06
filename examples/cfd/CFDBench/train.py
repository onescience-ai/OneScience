import os
import time
import torch
import numpy as np
from pathlib import Path
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm
from shutil import copyfile
from copy import deepcopy

# Onescience Imports
from onescience.utils.YParams import YParams
from onescience.distributed.manager import DistributedManager
from onescience.datapipes.cfd import CFDBenchDatapipe
from torch.nn.parallel import DistributedDataParallel as DDP

from onescience.utils.cfdbench.utils import (
    dump_json, plot_loss, plot_predictions, load_best_ckpt,get_output_dir
)

# Models (Static)
from onescience.models.cfdbench.deeponet import DeepONet
from onescience.models.cfdbench.ffn import FfnModel
from onescience.models.cfdbench.loss import loss_name_to_fn

def init_static_model(cfg, device):
    """
    专门初始化非自回归模型 (DeepONet, FFN)
    """
    model_cfg = cfg.model
    train_cfg = cfg.training
    data_cfg = cfg.datapipe.data
    source_cfg = cfg.datapipe.source

    loss_fn = loss_name_to_fn(train_cfg.loss_name)
    
    if "cylinder" in cfg.datapipe.source:
        # (density, viscosity, u_top, h, w, radius, center_x, center_y)
        n_case_params = 8
    else:
        n_case_params = 5  # (density, viscosity, u_top, h, w)
        
    query_coord_dim = 3 # (t, x, y) - Static models usually query spacetime

    if model_cfg.name == "deeponet":
        # Static DeepONet: Branch takes parameters, Trunk takes coordinates
        model = DeepONet(
            branch_dim=n_case_params,
            trunk_dim=query_coord_dim,
            loss_fn=loss_fn,
            width=model_cfg.deeponet_width,
            trunk_depth=model_cfg.trunk_depth,
            branch_depth=model_cfg.branch_depth,
            act_name=model_cfg.act_fn,
            act_norm=model_cfg.act_scale_invariant,
            act_on_output=model_cfg.act_on_output,
        )
    elif model_cfg.name == "ffn":
        widths = (
            [n_case_params + query_coord_dim]
            + [model_cfg.ffn_width] * model_cfg.ffn_depth
            + [1]
        )
        model = FfnModel(
            widths=widths,
            loss_fn=loss_fn,
        )
    else:
        raise ValueError(f"Invalid static model name: {model_cfg.name}")

    return model.to(device)

def evaluate(model, loader, output_dir, dist, batch_size, plot_interval=1, measure_time=False):
    """
    非自回归评估逻辑
    """
    model_eval = model.module if hasattr(model, "module") else model
    model_eval.eval()

    if measure_time:
        assert batch_size == 1

    scores = {name: [] for name in model_eval.loss_fn.get_score_names()}
    all_preds = []

    if dist.rank == 0:
        print(f"=== Evaluating (rank {dist.rank}) ===")
        print(f"# batches: {len(loader)}")

    start_time = time.time()
    
    with torch.no_grad():
        for step, batch in enumerate(tqdm(loader, disable=dist.rank!=0)):
            # Move to device
            batch = {k: v.to(dist.device) for k, v in batch.items()}
            
            case_params = batch["case_params"] # (b, n_params)
            t = batch["t"] # (b, 1)
            label = batch["label"] # (b, 2, h, w) or (b, c, h, w)

            height, width = label.shape[-2:]

            preds = model_eval.generate_one(
                case_params=case_params, t=t, height=height, width=width
            )
            
            loss = model_eval.loss_fn(labels=label[:, :1], preds=preds)
            for key in scores:
                scores[key].append(loss[key].item())

            preds_expanded = preds.repeat(1, 3, 1, 1) 
            all_preds.append(preds_expanded.cpu().detach())

            if dist.rank == 0 and step % plot_interval == 0 and not measure_time:
                image_dir = output_dir / "images"
                image_dir.mkdir(exist_ok=True, parents=True)
                plot_predictions(
                    inp=None, # Static models don't have "input fields" like auto models
                    label=label[0][0],
                    pred=preds_expanded[0][0],
                    out_dir=image_dir,
                    step=step,
                )

    if measure_time:
        if dist.rank == 0:
            print("Memory usage:")
            print(torch.cuda.memory_summary("cuda"))
            print("Time usage:")
            time_per_step = 1000 * (time.time() - start_time) / len(loader)
            print(f"Time per step: {time_per_step:.3f} ms")
        exit()

    # Summarize scores
    avg_scores = {key: np.mean(vals) for key, vals in scores.items()}
    if dist.rank == 0:
        for key, val in avg_scores.items():
            print(f"{key}: {val:.4e}")
        plot_loss(scores["nmse"], output_dir / "loss.png")

    return dict(
        scores=dict(mean=avg_scores, all=scores),
        preds=all_preds,
    )

def test(model, datapipe, output_dir, dist, plot_interval=10, measure_time=False):
    """
    测试逻辑
    """
    if dist.world_size > 1 and dist.rank != 0:
        return

    if dist.rank == 0:
        output_dir.mkdir(exist_ok=True, parents=True)
        print(f"=== Testing (rank {dist.rank}) ===")

    # Static testing usually done with batch_size=1
    test_loader = datapipe.test_dataloader()
    
    result = evaluate(
        model,
        test_loader,
        output_dir,
        dist,
        batch_size=1, 
        plot_interval=plot_interval,
        measure_time=measure_time
    )
    
    if dist.rank == 0:
        torch.save(result["preds"], output_dir / "preds.pt")
        dump_json(result["scores"], output_dir / "scores.json")
        print("=== Testing done ===")

def train(cfg, model, datapipe, output_dir, dist):
    """
    训练逻辑
    """
    train_cfg = cfg.training
    
    train_loader, train_sampler = datapipe.train_dataloader()
    val_loader, val_sampler = datapipe.val_dataloader()
    
    if dist.rank == 0:
        print("==== Training ====")
        print(f"Output dir: {output_dir}")
        print(f"# epoch: {train_cfg.num_epochs}")
        print(f"# GPUs: {dist.world_size}")

    optimizer = Adam(model.parameters(), lr=train_cfg.lr)
    scheduler = lr_scheduler.StepLR(
        optimizer, step_size=train_cfg.lr_step_size, gamma=train_cfg.lr_gamma
    )

    start_time = time.time()
    global_step = 0
    all_train_losses = []

    for ep in range(train_cfg.num_epochs):
        if train_sampler:
            train_sampler.set_epoch(ep)
            
        ep_start_time = time.time()
        ep_train_losses = []
        model.train()
        
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(dist.device) for k, v in batch.items()}
            
            outputs = model(**batch)
            losses = outputs["loss"]
            loss = losses[train_cfg.loss_name]

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ep_train_losses.append(loss.item())
            global_step += 1
            
            # Log
            if dist.rank == 0 and global_step % train_cfg.log_interval == 0:
                avg_loss = np.mean(ep_train_losses[-train_cfg.log_interval:])
                print(f"Ep {ep} | Step {step} | Loss: {avg_loss:.3e} | LR: {scheduler.get_last_lr()[0]:.3e}")

        scheduler.step()

        # Evaluate & Checkpoint (Rank 0 only)
        if dist.rank == 0 and (ep + 1) % train_cfg.eval_interval == 0:
            ckpt_dir = output_dir / f"ckpt-{ep}"
            ckpt_dir.mkdir(exist_ok=True, parents=True)
            
            # Evaluate
            dev_result = evaluate(
                model, 
                val_loader, 
                ckpt_dir, 
                dist, 
                batch_size=cfg.datapipe.dataloader.eval_batch_size
            )
            
            dev_scores = dev_result["scores"]
            dump_json(dev_scores, ckpt_dir / "dev_loss.json")
            dump_json(ep_train_losses, ckpt_dir / "train_loss.json")

            # Save Checkpoint
            model_to_save = model.module if hasattr(model, "module") else model
            ckpt_path = ckpt_dir / "model.pt"
            print(f"Saving checkpoint to {ckpt_path}")
            torch.save(model_to_save.state_dict(), ckpt_path)

            # Save Summary
            ep_scores = dict(
                ep=ep,
                train_loss=np.mean(ep_train_losses),
                dev_loss=np.mean(dev_scores["mean"]["nmse"]),
                time=time.time() - ep_start_time,
            )
            dump_json(ep_scores, ckpt_dir / "scores.json")

        if dist.world_size > 1:
            torch.distributed.barrier()
        
        all_train_losses.append(ep_train_losses)

    if dist.rank == 0:
        flat_losses = sum(all_train_losses, [])
        dump_json(flat_losses, output_dir / "train_losses.json")
        plot_loss(flat_losses, output_dir / "train_losses.png")
        print("Training Finished.")

def main():
    # Initialize
    DistributedManager.initialize()
    dist = DistributedManager()
    
    # Config
    config_path = "conf/cfdbench.yaml"
    cfg = YParams(config_path, "root")
    
    # 强制设置为 static 模式
    cfg.datapipe.data.task_type = "static"

    # 生成输出目录
    output_dir = get_output_dir(cfg, is_auto=False)
    
    if dist.rank == 0:
        print("#" * 80)
        print(f"Config loaded from {config_path}")
        print(f"Output Directory: {output_dir}")
        print("#" * 80)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Data
    if dist.rank == 0: print("Loading data...")
    datapipe = CFDBenchDatapipe(cfg.datapipe, distributed=(dist.world_size > 1))
    
    # Model
    if dist.rank == 0: print("Loading model...")
    model = init_static_model(cfg, dist.device)
    
    if dist.rank == 0:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model has {num_params} parameters")

    # DDP Wrapping
    if "train" in cfg.training.mode and dist.world_size > 1:
        model = DDP(model, device_ids=[dist.device])

    # Training Phase
    if "train" in cfg.training.mode:
        train(cfg, model, datapipe, output_dir, dist)

    # Testing Phase
    if "test" in cfg.training.mode and dist.rank == 0:
        print("Starting Testing Phase...")
        
        model_to_test = model.module if hasattr(model, "module") else model
        
        try:
            load_best_ckpt(model_to_test, output_dir)
        except Exception as e:
            print(f"Warning: Could not load best checkpoint: {e}. Testing with current weights.")

        test_dir = output_dir / "test"
        test_dir.mkdir(exist_ok=True)
        
        test(
            model=model_to_test,
            datapipe=datapipe,
            output_dir=test_dir,
            dist=dist,
            plot_interval=10
        )

    dist.cleanup()

if __name__ == "__main__":
    main()