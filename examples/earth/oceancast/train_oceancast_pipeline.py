"""
This Code is used to train pipeline_parallel mode of oceancast

"""

import logging
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from onescience.models.afno.afnonet_pipeline import build_pipeline_model
from onescience.utils.fcn.darcy_loss_ocean import LpLoss as loss_function
from onescience.utils.fcn.data_loader_ocean import get_data_loader
from onescience.utils.fcn.YParams import YParams


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger()

    current_path = os.getcwd()
    config_file_path = os.path.join(current_path, "conf/oceancast.yaml")
    params = YParams(config_file_path, "afno_backbone")

    world_rank = 0
    local_rank = 0
    params["world_size"] = int(os.environ.get("WORLD_SIZE", 1))
    if params["world_size"] > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ["LOCAL_RANK"])
        world_rank = dist.get_rank()
    else:
        world_rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    train_data_loader, train_dataset, train_sampler = get_data_loader(
        params, dist.is_initialized(), mode="train"
    )
    valid_data_loader, valid_dataset, valid_sampler = get_data_loader(
        params, dist.is_initialized(), mode="valid"
    )

    # ✅ 使用 Pipeline 模型替换原始 AFNONet 构建
    model = build_pipeline_model(params, num_devices=4, chunks=params.chunks)
    # model = model.to(local_rank)
    if params["world_size"] > 1:
        model = DistributedDataParallel(model)  # , find_unused_parameters=True

    first_stage_device = next(model.module[0].parameters()).device
    last_stage_device = next(model.module[-1].parameters()).device

    mask_tensor = torch.load(params.maskpath)
    mask = mask_tensor.to(last_stage_device)
    loss_obj = lambda x, y: loss_function(mask.to(y.device))(x, y)

    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    scheduler = get_scheduler(optimizer, params)

    best_valid_loss = 1.0e6
    best_loss_epoch = 0
    model_path = params.model_path
    os.makedirs(model_path, exist_ok=True)
    model_name = f"{model_path}/model_pp.pth"
    train_loss_file = f"{model_path}/trloss.npy"
    valid_loss_file = f"{model_path}/valoss.npy"
    if world_rank == 0:
        print(f"weights and loss are saved at {model_path}")

    train_losses = np.empty((0,), dtype=np.float32)
    valid_losses = np.empty((0,), dtype=np.float32)

    if world_rank == 0:
        logger.info(f"start training {params['name']}...")
    params["max_epochs"] = 1
    for epoch in range(params.max_epochs):
        t1 = time.perf_counter()
        if dist.is_initialized():
            train_sampler.set_epoch(epoch)
            valid_sampler.set_epoch(epoch)
        model.train()
        t1 = time.perf_counter()
        train_loss = 0
        for iteration_idx, data in enumerate(train_data_loader):
            inp = data[0].to(first_stage_device, dtype=torch.float)
            tar = data[1].to(last_stage_device, dtype=torch.float)
            gen = model(inp).local_value()  # ✅ Pipe 输出为 RRef
            loss = loss_obj(gen, tar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if dist.is_initialized():
                dist.all_reduce(loss)
                train_loss += float(loss / dist.get_world_size())
            else:
                train_loss += float(loss)
            if world_rank == 0 and iteration_idx % (len(train_data_loader) // 8) == 0:
                logger.info(
                    f"Epoch [{epoch + 1}/{params.max_epochs}], Train MiniBatch {iteration_idx}/{len(train_data_loader)} done..."
                )
        # if world_rank == 0:
        #     print(f'first epoch done, cost: {time.perf_counter() - t1: .2f}') # 202
        # exit()

        train_loss /= len(train_data_loader)

        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for iteration_idx, data in enumerate(valid_data_loader):
                inp = data[0].to(first_stage_device, dtype=torch.float)
                tar = data[1].to(last_stage_device, dtype=torch.float)
                gen = model(inp).local_value()  # ✅ Pipe 输出为 RRef
                loss = loss_obj(gen, tar)
                if dist.is_initialized():
                    dist.all_reduce(loss)
                    valid_loss += float(loss / dist.get_world_size())
                else:
                    valid_loss += float(loss)
                if (
                    world_rank == 0
                    and iteration_idx % (len(valid_data_loader) // 3) == 0
                ):
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
                f"Valid Loss: {valid_loss:.4f}, "
                f"Best loss at Epoch: {best_loss_epoch + 1}"
            )
            train_losses = np.append(train_losses, train_loss)
            valid_losses = np.append(valid_losses, valid_loss)
            np.save(train_loss_file, train_losses)
            np.save(valid_loss_file, valid_losses)

        if epoch - best_loss_epoch > params.patience:
            print("Loss has not decreased in 30 epochs, stopping training...")
            exit()
        if world_rank == 0:
            print("=" * 10, f"\n pp cost {time.perf_counter() - t1:.2f} \n", "=" * 10)


def save_checkpoint(
    model, optimizer, scheduler, best_valid_loss, best_loss_epoch, model_name
):
    model_to_save = model.module if hasattr(model, "module") else model
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
