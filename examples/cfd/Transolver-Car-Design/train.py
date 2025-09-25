import json
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

from onescience.distributed.manager import DistributedManager


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def train(device, model, train_loader, optimizer, scheduler, reg=1):
    model.train()
    criterion_func = nn.MSELoss(reduction="none")
    losses_press = []
    losses_velo = []
    for cfd_data, geom in train_loader:
        cfd_data = cfd_data.to(device)
        geom = geom.to(device)
        optimizer.zero_grad()
        out = model((cfd_data, geom))
        targets = cfd_data.y

        loss_press = criterion_func(
            out[cfd_data.surf, -1], targets[cfd_data.surf, -1]
        ).mean(dim=0)
        loss_velo_var = criterion_func(out[:, :-1], targets[:, :-1]).mean(dim=0)
        loss_velo = loss_velo_var.mean()
        total_loss = loss_velo + reg * loss_press

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        losses_press.append(loss_press.item())
        losses_velo.append(loss_velo.item())
    return np.mean(losses_press), np.mean(losses_velo)


@torch.no_grad()
def test(device, model, test_loader):
    model.eval()
    criterion_func = nn.MSELoss(reduction="none")
    losses_press = []
    losses_velo = []
    for cfd_data, geom in test_loader:
        cfd_data = cfd_data.to(device)
        geom = geom.to(device)
        out = model((cfd_data, geom))
        targets = cfd_data.y

        loss_press = criterion_func(
            out[cfd_data.surf, -1], targets[cfd_data.surf, -1]
        ).mean(dim=0)
        loss_velo_var = criterion_func(out[:, :-1], targets[:, :-1]).mean(dim=0)
        loss_velo = loss_velo_var.mean()

        losses_press.append(loss_press.item())
        losses_velo.append(loss_velo.item())
    return np.mean(losses_press), np.mean(losses_velo)


def main(
    device,
    train_dataset,
    val_dataset,
    model,
    hparams,
    path,
    reg=1,
    val_iter=1,
    coef_norm=[],
):
    dist = DistributedManager()
    model = model.to(device)
    if dist.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[dist.local_rank],
            output_device=dist.device,
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr"])
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=hparams["lr"],
        total_steps=(len(train_dataset) // hparams["batch_size"] + 1)
        * hparams["nb_epochs"],
        final_div_factor=1000.0,
    )
    start = time.time()

    if dist.world_size > 1:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        shuffle_flag = False
        num_workers = 1
    else:
        train_sampler = None
        shuffle_flag = True
        num_workers = 4
    train_loader = DataLoader(
        train_dataset,
        batch_size=hparams["batch_size"],
        sampler=train_sampler,
        shuffle=shuffle_flag,
        drop_last=True,
        num_workers=num_workers,
    )

    for epoch in range(hparams["nb_epochs"]):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        loss_press, loss_velo = train(
            device, model, train_loader, optimizer, lr_scheduler, reg=reg
        )
        train_loss = loss_velo + reg * loss_press

        log_msg = f"Epoch [{epoch+1}/{hparams['nb_epochs']}]"
        log_msg += f" - Train Loss: {train_loss:.4f}"

        if val_iter and (epoch == hparams["nb_epochs"] - 1 or epoch % val_iter == 0):
            val_loader = DataLoader(val_dataset, batch_size=1)
            loss_press_val, loss_velo_val = test(device, model, val_loader)
            val_loss = loss_velo_val + reg * loss_press_val
            del val_loader
            log_msg += f" - Val Loss: {val_loss:.4f}"

        if dist.rank == 0:
            logging.info(log_msg)

    end = time.time()
    time_elapsed = end - start
    params_model = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if dist.rank == 0:
        logging.info(f"Number of parameters: {params_model}")
        logging.info(f"Time elapsed: {time_elapsed:.2f} seconds")

        to_save = model.module if dist.world_size > 1 else model
        torch.save(to_save, os.path.join(path, f'model_{hparams["nb_epochs"]}.pth'))

    if val_iter and dist.rank == 0:
        log_data = {
            "nb_parameters": params_model,
            "time_elapsed": time_elapsed,
            "hparams": hparams,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "coef_norm": coef_norm,
        }
        with open(os.path.join(path, f'log_{hparams["nb_epochs"]}.json'), "w") as f:
            json.dump(log_data, f, indent=4, cls=NumpyEncoder)
    return model
