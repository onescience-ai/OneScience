import os
import sys
import torch
import torch.nn as nn
import numpy as np
import copy
from timeit import default_timer
from pathlib import Path
from tqdm import tqdm
import argparse
from onescience.utils.YParams import YParams
from onescience.distributed.manager import DistributedManager
from onescience.datapipes.cfd.PDENNEval import PDEBenchPINODatapipe
from torch.nn.parallel import DistributedDataParallel as DDP

from onescience.models.pdenneval.pino_fno import FNO1d, FNO2d, FNO3d
from onescience.utils.pdenneval.pino_loss import pde_loss
from onescience.utils.pdenneval.pino_utils import generate_input, to_device, count_params


def get_model(spatial_dim, if_temporal, cfg):
    assert spatial_dim <= 3, "Spatial dimension of data can not exceed 3."
    model_args = cfg.model
    initial_step = cfg.datapipe.data.initial_step
    dim = spatial_dim + 1 if if_temporal else spatial_dim
    in_dim = model_args.in_channels * initial_step + dim
    
    if dim == 1:
        model = FNO1d(
            in_dim=in_dim,
            out_dim=model_args.out_channels,
            modes=model_args.modes1,
            fc_dim=model_args.fc_dim,
            width=model_args.width,
            act=model_args.act
        )
    elif dim == 2:
        model = FNO2d(
            in_dim=in_dim,
            out_dim=model_args.out_channels,
            modes1=model_args.modes1,
            modes2=model_args.modes2,
            fc_dim=model_args.fc_dim,
            width=model_args.width,
            act=model_args.act
        )
    elif dim == 3:
        # 使用标准的 FNO3d
        model = FNO3d(
            in_dim=in_dim,
            out_dim=model_args.out_channels,
            modes1=model_args.modes1,
            modes2=model_args.modes2,
            modes3=model_args.modes3,
            fc_dim=model_args.fc_dim,
            width=model_args.width,
            act=model_args.act
        )
    else:
        raise NotImplementedError(f"Dimension {dim} is not supported.")
        
    return model
    
def train_loop(dataloader, pdeloader, model, if_temporal, optimizer, device, train_args, rank, pbar=None):
    model.train()

    data_weight = train_args.xy_loss
    f_weight = train_args.f_loss
    ic_loss_weight = train_args.ic_loss
    loss_fn = nn.MSELoss(reduction="mean")

    train_loss = 0.0

    loader_iter = iter(dataloader)
    pde_iter = iter(pdeloader)

    num_batches = len(dataloader) if hasattr(dataloader, "__len__") else None
    if num_batches is None:
        # 兜底：如果 dataloader 没有 __len__，就按迭代器跑
        num_batches = 0
        for _ in dataloader:
            num_batches += 1
        loader_iter = iter(dataloader)

    for _ in range(num_batches):
        # 1) Data batch
        try:
            a, u, grid = next(loader_iter)
        except StopIteration:
            loader_iter = iter(dataloader)
            a, u, grid = next(loader_iter)

        if torch.any(u.isnan()):
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(skip="nan")
            continue

        bs = a.shape[0]
        a, u, grid = to_device([a, u, grid], device)

        if not if_temporal:  # Darcy
            a_in = u[..., 0:1]
            u_in = u[..., 1:2]
            input_tensor = generate_input(a_in, grid[..., :-1]).squeeze(-2)
            u_target = u_in
        else:
            input_tensor = generate_input(a, grid)
            u_target = u

        pred = model(input_tensor)
        data_loss = loss_fn(pred[..., :u_target.shape[-1]].reshape(bs, -1), u_target.reshape(bs, -1))

        # 2) PDE loss
        if f_weight > 0:
            try:
                a_pde, u_pde, grid_pde = next(pde_iter)
            except StopIteration:
                pde_iter = iter(pdeloader)
                a_pde, u_pde, grid_pde = next(pde_iter)

            a_pde, u_pde, grid_pde = to_device([a_pde, u_pde, grid_pde], device)

            if not if_temporal:
                a_in_pde = u_pde[..., 0:1]
                u_in_pde = u_pde[..., 1:2]
                input_pde = generate_input(a_in_pde, grid_pde[..., :-1]).squeeze(-2)
            else:
                input_pde = generate_input(a_pde, grid_pde)

            pred_pde = model(input_pde)

            ic_loss_val, f_loss_val = pde_loss(
                pred_pde,
                a_in_pde if not if_temporal else a_pde,
                u_in_pde if not if_temporal else u_pde,
                train_args,
                grid_pde
            )
        else:
            ic_loss_val = torch.tensor(0.0, device=device)
            f_loss_val = torch.tensor(0.0, device=device)

        loss = data_weight * data_loss + ic_loss_weight * ic_loss_val + f_weight * f_loss_val

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val = float(loss.item())
        train_loss += loss_val

        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(
                loss=f"{loss_val:.3e}",
                data=f"{float(data_loss.item()):.3e}",
                f=f"{float(f_loss_val.item()):.3e}",
                ic=f"{float(ic_loss_val.item()):.3e}",
            )

    return train_loss / num_batches


@torch.no_grad()
def val_loop(dataloader, model, if_temporal, device):
    model.eval()
    loss_fn = nn.MSELoss(reduction="mean")
    val_loss = 0.0
    
    for a, u, grid in dataloader:
        bs = a.shape[0]
        a, u, grid = to_device([a, u, grid], device)
        
        if not if_temporal:
            a_in = u[..., 0:1]
            u_in = u[..., 1:2]
            input_tensor = generate_input(a_in, grid[..., :-1]).squeeze(-2)
            u = u_in
        else:
            input_tensor = generate_input(a, grid)
            
        pred = model(input_tensor)
        val_loss += loss_fn(pred[..., :u.shape[-1]].reshape(bs, -1), u.reshape(bs, -1)).item()
        
    return val_loss / len(dataloader)

def main():
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device
    
    parser = argparse.ArgumentParser(description='Train PINO model')
    parser.add_argument('config', type=str, help='Path to config file')
    args = parser.parse_args()
    config_path = args.config
    # 验证配置文件是否存在
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
        
    cfg = YParams(config_path, "pino_config")
    
    output_dir = Path(cfg.training.output_dir)
    if dist.rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output: {output_dir}")

    # Datapipe
    datapipe = PDEBenchPINODatapipe(cfg, distributed=(dist.world_size > 1))
    train_loader, train_sampler = datapipe.train_dataloader()
    pde_loader, pde_sampler = datapipe.pde_dataloader()
    val_loader, val_sampler = datapipe.val_dataloader()
    

    train_args_dict = cfg.training.to_dict()
    train_args_dict['dx'] = datapipe.dx
    train_args_dict['dt'] = datapipe.dt
    # Wrap back to object for compatibility if pde_loss expects object access (args.dx)
    # Or modify pde_loss to accept dict. Assuming dict/namespace wrapper.
    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super(AttrDict, self).__init__(*args, **kwargs)
            self.__dict__ = self
    train_args_obj = AttrDict(train_args_dict)

    # Determine Spatial Dim
    temp_sample = next(iter(val_loader))
    _, sample_u, _ = temp_sample
    spatial_dim = len(sample_u.shape) - 3
    if_temporal = True if sample_u.shape[-2] != 1 else False

    # Model
    model = get_model(spatial_dim, if_temporal, cfg).to(device)
    if dist.world_size > 1:
        model = DDP(model, device_ids=[dist.local_rank], output_device=dist.local_rank)
        
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.optimizer.lr, weight_decay=cfg.training.optimizer.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.training.scheduler.step_size, gamma=cfg.training.scheduler.gamma)
    
    if dist.rank == 0:
        print("Starting Training...")
    
    for epoch in range(cfg.training.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        if pde_sampler:
            pde_sampler.set_epoch(epoch)
    
        pbar = None
        if dist.rank == 0:
            total = len(train_loader) if hasattr(train_loader, "__len__") else None
            pbar = tqdm(
                total=total,
                desc=f"Epoch {epoch+1}/{cfg.training.epochs}",
                dynamic_ncols=True,
                leave=False,
                mininterval=0.5,
            )
    
        train_loss = train_loop(
            train_loader, pde_loader, model, if_temporal, optimizer,
            device, train_args_obj, dist.rank, pbar=pbar
        )
    
        if pbar is not None:
            pbar.close()
    
        scheduler.step()
    
        if dist.rank == 0:
            tqdm.write(f"[Epoch {epoch}] Train Loss: {train_loss:.4e}")
    
            if (epoch + 1) % cfg.training.save_period == 0:
                val_loss = val_loop(val_loader, model, if_temporal, device)
                tqdm.write(f"[Epoch {epoch}] Val Loss: {val_loss:.4e}")
    
                ckpt_path = output_dir / f"model_epoch_{epoch}.pt"
                model_state = model.module.state_dict() if dist.world_size > 1 else model.state_dict()
                torch.save(model_state, ckpt_path)


    dist.cleanup()

if __name__ == "__main__":
    main()