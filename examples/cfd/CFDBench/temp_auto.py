import os
import time
import torch
import numpy as np
from pathlib import Path
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm
from shutil import copyfile

# Onescience imports
from onescience.utils.YParams import YParams
from onescience.distributed.manager import DistributedManager
from onescience.datapipes import CFDBenchDatapipe
from torch.nn.parallel import DistributedDataParallel as DDP

# Utils from existing code (Assuming they exist)
from onescience.utils.cfdbench.utils import (
    dump_json, plot_loss, plot_predictions, load_best_ckpt
)
# Model Factory (Assuming you keep this or refactor similarly)
from onescience.utils.cfdbench.utils_auto import init_model 

def evaluate(model, loader, output_dir, dist):
    model.eval()
    scores = {} 
    # Logic simplified for brevity, similar to original evaluate
    # ... (Evaluation loop implementation) ...
    return {"scores": {"nmse": 0.0}} # Placeholder return

def main():
    # 1. Initialize
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device
    
    # 2. Config
    config_path = "conf/cfdbench.yaml"
    cfg = YParams(config_path, "training")
    cfg_data = YParams(config_path, "datapipe")
    cfg_model = YParams(config_path, "model")
    
    # Ensure task type is auto
    cfg_data.data.task_type = "auto"

    if dist.rank == 0:
        print(f"Loading Config from {config_path}")
        output_dir = Path(cfg.output_dir) / cfg_data.source.data_name / "auto"
        output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Data
    datapipe = CFDBenchDatapipe(cfg_data, distributed=(dist.world_size > 1))
    train_loader, train_sampler = datapipe.train_dataloader()
    val_loader, val_sampler = datapipe.val_dataloader()

    # 4. Model
    # We create a dummy 'args' object because init_model expects it, 
    # or you can refactor init_model to accept YParams
    class AdapterArgs:
        def __init__(self, **entries): self.__dict__.update(entries)
    
    model_args = AdapterArgs(**cfg_model.to_dict())
    model_args.model = cfg_model.name # Ensure name match
    
    model = init_model(model_args).to(device)
    
    if dist.world_size > 1:
        model = DDP(model, device_ids=[dist.local_rank], output_device=dist.local_rank)

    # 5. Optimizer
    optimizer = Adam(model.parameters(), lr=cfg.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma)

    # 6. Train Loop
    if dist.rank == 0:
        print("Starting Auto-regressive Training...")

    for epoch in range(cfg.num_epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
            
        model.train()
        ep_losses = []
        
        for step, batch in enumerate(train_loader):
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch) # expects inputs, label, mask, case_params
            loss_dict = outputs["loss"]
            loss = loss_dict[cfg.loss_name]
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            ep_losses.append(loss.item())
            
            if dist.rank == 0 and step % cfg.log_interval == 0:
                print(f"Ep {epoch} | Step {step} | Loss: {loss.item():.4e}")

        scheduler.step()
        
        # Evaluation & Saving (Rank 0)
        if dist.rank == 0 and (epoch + 1) % cfg.eval_interval == 0:
            avg_loss = np.mean(ep_losses)
            print(f"Epoch {epoch} Done. Avg Train Loss: {avg_loss:.4e}")
            
            # Save checkpoint
            ckpt_path = output_dir / "model_latest.pt"
            torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(), ckpt_path)

    dist.cleanup()

if __name__ == "__main__":
    main()