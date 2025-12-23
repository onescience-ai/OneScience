import os
import torch
import torch.nn as nn
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm

from onescience.utils.YParams import YParams
from onescience.distributed.manager import DistributedManager
from onescience.datapipes import PDEBenchMPNNDatapipe
from torch.nn.parallel import DistributedDataParallel as DDP

# MPNN Model
from onescience.models.pdenneval.mpnn import MPNN

def get_model(cfg, pde):
    # MPNN is often a list of models (one per output var)
    model = nn.ModuleList([
        MPNN(
            pde=pde,
            time_window=cfg.datapipe.data.time_window,
            hidden_features=cfg.model.hidden_features,
            hidden_layers=cfg.model.hidden_layer,
            eq_variables=cfg.datapipe.data.variables
        ) for _ in range(cfg.model.num_outputs)
    ])
    return model

def train_loop(dataloader, model, loss_fn, optimizer, device, graph_creator, cfg, rank, pbar=None):
    model.train()
    losses = []

    max_unrolling = cfg.training.unrolling
    unrolling_opts = [r for r in range(max_unrolling + 1)]

    for u, x, variables in dataloader:
        u, x = u.to(device), x.to(device)
        batch_size = u.shape[0]
        num_outputs = u.shape[-1]

        unrolled_graphs = random.choice(unrolling_opts)

        max_start = graph_creator.nt - graph_creator.tw - (graph_creator.tw * unrolled_graphs)
        if max_start <= graph_creator.tw:
            if pbar is not None:
                pbar.update(1)
            continue

        steps = [t for t in range(graph_creator.tw, max_start + 1)]
        random_steps = random.choices(steps, k=batch_size)

        data, labels = graph_creator.create_data(u, random_steps)
        graph = graph_creator.create_graph(data, labels, x, variables, random_steps).to(device)

        with torch.no_grad():
            for _ in range(unrolled_graphs):
                pred = torch.empty_like(graph.y)
                for i in range(num_outputs):
                    pred[..., i] = model[i](graph, i)

                random_steps = [rs + graph_creator.tw for rs in random_steps]
                _, labels = graph_creator.create_data(u, random_steps)
                graph = graph_creator.create_next_graph(graph, pred, labels, random_steps).to(device)

        pred = torch.empty_like(graph.y)
        for i in range(num_outputs):
            pred[..., i] = model[i](graph, i)

        loss = loss_fn(pred, graph.y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.3e}")

    return float(np.mean(losses)) if len(losses) else 0.0


@torch.no_grad()
def val_loop(dataloader, model, loss_fn, device, graph_creator):
    model.eval()
    losses = []
    
    for u, x, variables in dataloader:
        u, x = u.to(device), x.to(device)
        batch_size = u.shape[0]
        num_outputs = u.shape[-1]
        
        # Validation usually starts at first window
        steps = [graph_creator.tw] * batch_size
        
        data, labels = graph_creator.create_data(u, steps)
        graph = graph_creator.create_graph(data, labels, x, variables, steps).to(device)
        
        # Rollout full trajectory
        target, pred = torch.Tensor().to(device), torch.Tensor().to(device)
        target = torch.cat((target, graph.y), dim=-2)
        
        # First step
        output = torch.empty_like(graph.y)
        for i in range(num_outputs):
            output[..., i] = model[i](graph, i)
        pred = torch.cat((pred, output), dim=-2)
        
        # Subsequent steps
        # Note: Logic follows original code's full rollout
        for step in range(2*graph_creator.tw, graph_creator.nt - graph_creator.tw + 1, graph_creator.tw):
            steps = [step] * batch_size
            _, labels = graph_creator.create_data(u, steps)
            graph = graph_creator.create_next_graph(graph, output, labels, steps).to(device)
            target = torch.cat((target, graph.y), dim=-2)
            
            output = torch.empty_like(graph.y)
            for i in range(num_outputs):
                output[..., i] = model[i](graph, i)
            pred = torch.cat((pred, output), dim=-2)
            
        losses.append(loss_fn(pred, target).item())
        
    return np.mean(losses)

def main():
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device
    
    cfg = YParams("config/config_1D_Diffusion-Reaction.yaml", "mpnn_config")
    output_dir = Path(cfg.training.output_dir)
    
    if dist.rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output: {output_dir}")

    # Datapipe
    datapipe = PDEBenchMPNNDatapipe(cfg, distributed=(dist.world_size > 1))
    train_loader, train_sampler = datapipe.train_dataloader()
    val_loader, val_sampler = datapipe.val_dataloader()
    
    graph_creator = datapipe.graph_creator.to(device) # GraphCreator has buffers?
    
    # Model
    model = get_model(cfg, datapipe.pde).to(device)
    if dist.world_size > 1:
        # MPNN is a ModuleList, DDP might need wrapping the whole list or individual modules
        # Usually wrapping the ModuleList container works
        model = DDP(model, device_ids=[dist.local_rank], output_device=dist.local_rank)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.optimizer.lr, weight_decay=cfg.training.optimizer.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.training.scheduler.step_size, gamma=cfg.training.scheduler.gamma)
    loss_fn = nn.MSELoss()
    
    if dist.rank == 0: print("Starting Training...")
    
    for epoch in range(cfg.training.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
    
        pbar = None
        if dist.rank == 0:
            # leave=False：每个 epoch 结束自动清掉，不堆叠一堆历史进度条
            pbar = tqdm(
                total=len(train_loader),
                desc=f"Epoch {epoch+1}/{cfg.training.epochs}",
                dynamic_ncols=True,
                leave=False,
                mininterval=0.5,   # 降低刷新频率，日志更干净
            )
    
        train_loss = train_loop(
            train_loader, model, loss_fn, optimizer, device,
            graph_creator, cfg, dist.rank, pbar=pbar
        )
    
        if pbar is not None:
            pbar.close()
    
        scheduler.step()
    
        if dist.rank == 0:
            tqdm.write(f"[Epoch {epoch}] Train Loss: {train_loss:.4e}")
    
            if (epoch + 1) % cfg.training.save_period == 0:
                val_loss = val_loop(val_loader, model, loss_fn, device, graph_creator)
                tqdm.write(f"[Epoch {epoch}] Val Loss: {val_loss:.4e}")
    
                ckpt_path = output_dir / f"model_epoch_{epoch}.pt"
                model_to_save = model.module if hasattr(model, "module") else model
                torch.save(model_to_save.state_dict(), ckpt_path)

    dist.cleanup()

if __name__ == "__main__":
    main()