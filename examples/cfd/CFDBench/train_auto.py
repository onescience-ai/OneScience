import os
import time
import torch
import numpy as np
from pathlib import Path
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm
from shutil import copyfile
from copy import deepcopy
# Onescience imports
from onescience.utils.YParams import YParams
from onescience.distributed.manager import DistributedManager
from onescience.datapipes.cfd import CFDBenchDatapipe
from torch.nn.parallel import DistributedDataParallel as DDP

from onescience.utils.cfdbench.utils import (
    dump_json, plot_loss, plot_predictions, load_best_ckpt, get_output_dir
)
from onescience.utils.cfdbench.utils_auto import init_model 

def evaluate(model, loader, output_dir, dist, batch_size, plot_interval=1):
    """
    评估逻辑
    """
    model_eval = model.module if hasattr(model, "module") else model
    model_eval.eval()
    
    # 获取评分指标名称
    scores = {name: [] for name in model_eval.loss_fn.get_score_names()}
    input_scores = deepcopy(scores)
    all_preds = []
    
    if dist.rank == 0:
        print(f"=== Evaluating (rank {dist.rank}) ===")
        print(f"# batches: {len(loader)}")
    
    with torch.no_grad():
        for step, batch in enumerate(tqdm(loader, disable=dist.rank!=0)):
            # Move data to GPU
            batch = {k: v.to(dist.device) for k, v in batch.items()}
            inputs = batch["inputs"] # (b, 2, h, w)
            labels = batch["label"] # (b, 2, h, w)
            
            # 1. Compute Input Loss (Baseline)
            input_loss = model_eval.loss_fn(
                labels=labels[:, :1], preds=inputs[:, :1]
            )
            for key in input_scores:
                input_scores[key].append(input_loss[key].cpu().tolist())
                
            # 2. Compute Prediction Loss
            outputs = model_eval(**batch)
            loss = outputs["loss"]
            preds = outputs["preds"] # Flattened or structured
            
            height, width = labels.shape[2:]
            preds = preds.view(-1, 1, height, width) # Ensure shape (b, 1, h, w)
            
            for key in scores:
                scores[key].append(loss[key].cpu().tolist())
            
            all_preds.append(preds.cpu().detach())
            
            # 3. Visualization
            if dist.rank == 0 and step % plot_interval == 0:
                image_dir = output_dir / "images"
                image_dir.mkdir(exist_ok=True, parents=True)
                plot_predictions(
                    inp=inputs[0][0],
                    label=labels[0][0],
                    pred=preds[0][0],
                    out_dir=image_dir,
                    step=step,
                )

    # Summarize scores
    avg_scores = {}
    for key in scores:
        mean = np.mean(scores[key])
        input_mean = np.mean(input_scores[key])
        avg_scores[key] = mean
        avg_scores[f"input_{key}"] = input_mean
        if dist.rank == 0:
            print(f"Prediction {key}: {mean:.4e} | Input {key}: {input_mean:.4e}")

    if dist.rank == 0:
        plot_loss(scores["nmse"], output_dir / "loss.png")
        
    return dict(
        preds=torch.cat(all_preds, dim=0),
        scores=dict(mean=avg_scores, all=scores),
    )

def test(model, datapipe, output_dir, dist, infer_steps=200, plot_interval=10):
    """
    测试逻辑
    """
    if dist.world_size > 1 and dist.rank != 0:
        return

    if dist.rank == 0:
        output_dir.mkdir(exist_ok=True, parents=True)
        print(f"=== Testing (rank {dist.rank}) ===")
        print(f"Plot interval: {plot_interval}")
        
    test_loader = datapipe.test_dataloader()
    
    result = evaluate(
        model,
        test_loader,
        output_dir=output_dir,
        dist=dist,
        batch_size=1, # Test always uses 1
        plot_interval=plot_interval
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
        print("====== Training ======")
        print(f"# epoch: {train_cfg.num_epochs}")
        print(f"# GPUs: {dist.world_size}")

    optimizer = Adam(model.parameters(), lr=train_cfg.lr)
    scheduler = lr_scheduler.StepLR(
        optimizer, step_size=train_cfg.lr_step_size, gamma=train_cfg.lr_gamma
    )

    start_time = time.time()
    global_step = 0
    train_losses = []

    for ep in range(train_cfg.num_epochs):
        if train_sampler:
            train_sampler.set_epoch(ep)
            
        ep_start_time = time.time()
        ep_train_losses = []
        model.train()
        
        for step, batch in enumerate(train_loader):
            # Move to device
            batch = {k: v.to(dist.device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss_dict = outputs["loss"]
            loss = loss_dict[train_cfg.loss_name] # usually "nmse"
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            ep_train_losses.append(loss.item())
            global_step += 1
            
            if dist.rank == 0 and global_step % train_cfg.log_interval == 0:
                print(f"Ep {ep} | Step {step} | Loss: {loss.item():.3e} | LR: {scheduler.get_last_lr()[0]:.3e}")

        scheduler.step()
        train_losses += ep_train_losses
        
        # Evaluate & Checkpoint
        if dist.rank == 0 and (ep + 1) % train_cfg.eval_interval == 0:
            ckpt_dir = output_dir / f"ckpt-{ep}"
            ckpt_dir.mkdir(exist_ok=True, parents=True)
            
            # Run Evaluation
            result = evaluate(
                model, 
                val_loader, 
                ckpt_dir, 
                dist, 
                batch_size=cfg.datapipe.dataloader.eval_batch_size
            )
            
            dev_scores = result["scores"]
            dump_json(dev_scores, ckpt_dir / "dev_scores.json")
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
                time=time.time() - ep_start_time
            )
            dump_json(ep_scores, ckpt_dir / "scores.json")
        
        if dist.world_size > 1:
            torch.distributed.barrier()

    if dist.rank == 0:
        print("====== Training done ======")
        dump_json(train_losses, output_dir / "train_losses.json")
        plot_loss(train_losses, output_dir / "train_losses.png")

def main():
    # 初始化
    DistributedManager.initialize()
    dist = DistributedManager()
    
    # 配置加载
    config_path = "conf/cfdbench.yaml"
    # Load entire config object passed to helper functions
    cfg = YParams(config_path, "root") 
    
    # 强制设置 task_type
    cfg.datapipe.data.task_type = "auto"

    # 生成输出目录 (使用新重构的 get_output_dir)
    output_dir = get_output_dir(cfg, is_auto=True)
    
    if dist.rank == 0:
        print("#" * 80)
        print(f"Config loaded from {config_path}")
        print(f"Output Directory: {output_dir}")
        print("#" * 80)
        output_dir.mkdir(parents=True, exist_ok=True)
        # 保存 YAML 也可以，或者保存解析后的字典
        # cfg.save(str(output_dir / "config.yaml")) 

    # 数据加载
    if dist.rank == 0: print("Loading data...")
    datapipe = CFDBenchDatapipe(cfg.datapipe, distributed=(dist.world_size > 1))
    
    # 模型初始化
    if dist.rank == 0: print("Loading model...")
    model = init_model(cfg)
    
    if dist.rank == 0:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model has {num_params} parameters")

    model = model.to(dist.device)
    
    # DDP Wrapping
    if "train" in cfg.training.mode and dist.world_size > 1:
        model = DDP(model, device_ids=[dist.device])

    # Training Phase
    if "train" in cfg.training.mode:
        train(cfg, model, datapipe, output_dir, dist)

    # Testing Phase
    if "test" in cfg.training.mode and dist.rank == 0:
        print("Starting Testing Phase...")
        
        # Unwrap model for testing
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
            infer_steps=20, # Hardcoded or add to config
            plot_interval=10 # Hardcoded or add to config
        )

if __name__ == "__main__":
    main()