import os
import sys
import time
import torch
import logging
import numpy as np
import torch.distributed as dist

from onescience.models.pangu import Pangu
from onescience.utils.YParams import YParams
from onescience.datapipes.climate import ERA5Datapipe
from onescience.metrics import L1_loss
from onescience.optimizers import FusedAdam
from onescience.memory.checkpoint import replace_function

def main():

    ## Logger init
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger()

    ## Device init
    device = 0

    ## Config init
    config_file_path = os.path.join(current_path, "conf/config.yaml")
    cfg = YParams(config_file_path, "model")
    cfg_data = YParams(config_file_path, "datapipe")

    ## DataLoader init
    datapipe = ERA5Datapipe(params=cfg_data, distributed=dist.is_initialized())
    train_dataloader, train_sampler = datapipe.train_dataloader()

    land_mask = torch.from_numpy(np.load(os.path.join(cfg_data.dataset.static_dir, "land_mask.npy")).astype(np.float32))
    soil_type = torch.from_numpy(np.load(os.path.join(cfg_data.dataset.static_dir, "soil_type.npy")).astype(np.float32))
    topography = torch.from_numpy(np.load(os.path.join(cfg_data.dataset.static_dir, "topography.npy")).astype(np.float32))
    topography = (topography - topography.mean()) / (topography.std(unbiased=False) + 1e-6)
    surface_mask = torch.stack([land_mask, soil_type, topography], dim=0).to(device)
    surface_mask = surface_mask.unsqueeze(0).repeat(cfg_data.dataloader.batch_size, 1, 1, 1)

    ## Model init
    pangu_model = Pangu(
        img_size=[721, 1440],
        patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        window_size=cfg.window_size,
    ).to(device)

    ## Optimizer init
    optimizer = FusedAdam(pangu_model.parameters(), betas=(0.9, 0.999), lr=5e-4, weight_decay=3e-6)
   
    logger.info(f"start training ...")

    for epoch in range(cfg.max_epoch):

        pangu_model.train()
        batch_start_time = time.time()
        for j, data in enumerate(train_dataloader):
            invar, outvar = data[0], data[1]
            invar_surface = invar[:, :4, :, :].to(device, dtype=torch.float32)
            invar_upper_air = invar[:, 4:, :, :].to(device, dtype=torch.float32)
            invar = torch.concat([invar_surface, surface_mask, invar_upper_air], dim=1)

            tar_surface = outvar[:, :4, :, :].to(device, dtype=torch.float32)
            tar_upper_air = outvar[:, 4:, :, :].to(device, dtype=torch.float32)

            with replace_function(
                pangu_model,
                ["layer1", "layer2", "layer3", "layer4"],
            ):
                out_surface, out_upper_air = pangu_model(invar)

            out_upper_air = out_upper_air.reshape(tar_upper_air.shape)

            loss1 = L1_loss(tar_surface, out_surface)
            loss2 = L1_loss(tar_upper_air, out_upper_air)
            loss = loss1 * 0.25 + loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if j % 5 == 0:
                batch_time = time.time() - batch_start_time
                logger.info(
                    f"Epoch [{epoch + 1}/{cfg.max_epoch}], Train MiniBatch {j}/{len(train_dataloader)} done, "
                    f"MiniBatch Time: {batch_time:.2f}s, Current Loss: {loss.item():.4f}"
                )
                batch_start_time = time.time()

if __name__ == "__main__":
    current_path = os.getcwd()
    sys.path.append(current_path)
    main()
