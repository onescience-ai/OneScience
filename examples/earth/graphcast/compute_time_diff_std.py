import torch
import os
import sys
import numpy as np
from tqdm import tqdm
from onescience.datapipes.climate import ERA5Datapipe
from onescience.utils.YParams import YParams


def main():
    # instantiate the training datapipe
    config_file_path = os.path.join(current_path, 'conf/config.yaml')
    cfg_data = YParams(config_file_path, "datapipe")
    datapipe = ERA5Datapipe(
        dataset_dir=cfg_data.dataset.data_dir,
        used_variables=cfg_data.dataset.channels,
        used_years=cfg_data.dataset.train_time,
        distributed=False
    )
    train_dataloader, train_sampler = datapipe.get_dataloader("train")
    
    print(f"Loaded training datapipe of length {len(train_dataloader)}")

    area = torch.abs(torch.cos(torch.linspace(-90, 90, steps=cfg_data.dataset.img_size[0]) * np.pi / 180))
    area /= torch.mean(area)
    area = area.unsqueeze(1)

    mean, mean_sqr = 0, 0
    for data in tqdm(train_dataloader):
        invar = data[0]  # [b, N, h, w]
        outvar = data[1]  # [b, N, h, w]
        diff = outvar - invar
        weighted_diff = area * diff
        weighted_diff_sqr = torch.square(weighted_diff)
        mean += torch.mean(weighted_diff, dim=(2, 3)) / len(train_dataloader)
        mean_sqr += torch.mean(weighted_diff_sqr, dim=(2, 3)) / len(train_dataloader)

    variance = mean_sqr - mean**2  # [1,num_channel, 1,1]
    std = torch.sqrt(variance)

    np.save("time_diff_std.npy", std.numpy())
    print(f"saving time_diff_std.npy, shapes are {std.numpy().shape}")


if __name__ == "__main__":
    current_path = os.getcwd()
    sys.path.append(current_path)
    main()