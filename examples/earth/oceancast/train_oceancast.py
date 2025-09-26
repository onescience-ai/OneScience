"""
This Code is adapted from NVIDIA-FourCastNet and is designed to achieve high-precision forecasting of ocean wave parameters (or sea surface parameters) with fine spatiotemporal resolution.
Change the conf/oceancast.yaml to determine the output.

1. Data Preparation:
   - Ensure all necessary data is downloaded before training the model.
   - Split the data into three parts: training, validation, and testing.
   - Save the data in the designated folder as specified in the `README.md`.

2. Data Organization:
   - Organize the data structure strictly following the guidelines provided in `README.md`.
   - If the data is stored in a different directory, update the `data_path` field in `conf/oceancast.yaml` accordingly.

"""

import logging
import os
import sys

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from onescience.models.afno.afnonet_ocean import AFNONet
from onescience.utils.fcn.darcy_loss_ocean import LpLoss as loss_function
from onescience.utils.fcn.data_loader_ocean import get_data_loader
from onescience.utils.fcn.YParams import YParams


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger()
    config_file_path = os.path.join(
        current_path, "conf/oceancast.yaml")
    params = YParams(config_file_path, "afno_backbone")
    params["world_size"] = 1
    if "WORLD_SIZE" in os.environ:
        params["world_size"] = int(os.environ["WORLD_SIZE"])

    world_rank = 0
    local_rank = 0

    if params["world_size"] > 1:
        dist.init_process_group(
            backend="nccl", init_method="env://")
        local_rank = int(os.environ["LOCAL_RANK"])
        world_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)

    train_data_loader, train_dataset, train_sampler = get_data_loader(
        params, dist.is_initialized(), mode="train"
    )
    valid_data_loader, valid_dataset, valid_sampler = get_data_loader(
        params, dist.is_initialized(), mode="valid"
    )

    model = AFNONet(params, drop_rate=0.3).to(local_rank)

    if params["world_size"] > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    mask_tensor = torch.load(params.maskpath)
    mask = mask_tensor.to(local_rank)
    loss_obj = loss_function(mask)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params.lr)
    scheduler = get_scheduler(optimizer, params)
    best_valid_loss = 1.0e6
    best_loss_epoch = 0
    model_path = params.model_path
    os.makedirs(model_path, exist_ok=True)
    model_name = f"{model_path}/model.pth"
    train_loss_file = f"{model_path}/trloss.npy"
    valid_loss_file = f"{model_path}/valoss.npy"
    if world_rank == 0:
        print(f"weights and loss are saved at {model_path}")

    train_losses = np.empty((0,), dtype=np.float32)
    valid_losses = np.empty((0,), dtype=np.float32)
    if world_rank == 0:
        logger.info(f"start training {params['name']}...")

    for epoch in range(params.max_epochs):
        if dist.is_initialized():
            train_sampler.set_epoch(epoch)
            valid_sampler.set_epoch(epoch)
        model.train()
        train_loss = 0
        for iteration_idx, data in enumerate(train_data_loader, 0):
            inp, tar = map(lambda x: x.to(
                local_rank, dtype=torch.float), data)
            gen = model(inp)
            loss = loss_obj(gen, tar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if params["world_size"] > 1:
                dist.all_reduce(loss)
                train_loss += float(loss /
                                    dist.get_world_size())
            else:
                train_loss += float(loss)
            if world_rank == 0:
                if iteration_idx % (len(train_data_loader) // 3) == 0:
                    logger.info(
                        f"Epoch [{epoch + 1}/{params.max_epochs}], Train MiniBatch {iteration_idx}/{len(train_data_loader)} done..."
                    )

        train_loss /= len(train_data_loader)
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for iteration_idx, data in enumerate(valid_data_loader, 0):
                inp, tar = map(lambda x: x.to(
                    local_rank, dtype=torch.float), data)
                gen = model(inp)
                loss = loss_obj(gen, tar)
                if params["world_size"] > 1:
                    dist.all_reduce(loss)
                    valid_loss += float(loss /
                                        dist.get_world_size())
                else:
                    valid_loss += float(loss)
                if world_rank == 0:
                    if iteration_idx % (len(valid_data_loader) // 3) == 0:
                        logger.info(
                            f"Epoch [{epoch + 1}/{params.max_epochs}], Valid MiniBatch {iteration_idx}/{len(valid_data_loader)} done..."
                        )
        valid_loss /= len(valid_data_loader)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_loss_epoch = epoch
            if world_rank == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    best_valid_loss,
                    best_loss_epoch,
                    model_name,
                )

        adjust_learning_rate(scheduler, valid_loss, params)

        if world_rank == 0:
            logger.info(
                f"Epoch [{epoch + 1}/{params.max_epochs}], "
                f"Train Loss: {train_loss:.4f}, "
                f"Valid Loss: {valid_loss:.4f},"
                f"Best loss at Epoch: {best_loss_epoch + 1}"
            )
            train_losses = np.append(
                train_losses, train_loss)
            valid_losses = np.append(
                valid_losses, valid_loss)

            np.save(train_loss_file, train_losses)
            np.save(valid_loss_file, valid_losses)

        if epoch - best_loss_epoch > params.patience:
            print(
                "Loss has not decrease in 30 epochs, stopping training...")
            exit()


def save_checkpoint(
    model, optimizer, scheduler, best_valid_loss, best_loss_epoch, model_name
):
    model_to_save = model.module if hasattr(
        model, "module") else model
    state = {
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_valid_loss": best_valid_loss,
        "best_loss_epoch": best_loss_epoch,
    }
    torch.save(state, model_name)


def get_scheduler(optimizer, params):
    if params.scheduler == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.2, patience=5, mode="min"
        )
    elif params.scheduler == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.T_max)
    else:
        return None


def adjust_learning_rate(scheduler, valid_loss, params):
    if params.scheduler == "ReduceLROnPlateau":
        scheduler.step(valid_loss)
    elif params.scheduler == "CosineAnnealingLR":
        scheduler.step()


if __name__ == "__main__":
    current_path = os.getcwd()
    sys.path.append(current_path)
    main()
