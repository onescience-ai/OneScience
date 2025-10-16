import torch
import os
import sys
import numpy as np

from onescience.datapipes.climate import ERA5HDF5Datapipe
from onescience.utils.fcn.YParams import YParams


def main():
    # instantiate the training datapipe
    config_file_path = os.path.join(current_path, 'conf/config.yaml')
    cfg = YParams(config_file_path, 'model')
    train_dataset = ERA5HDF5Datapipe(params=cfg, distributed=False)
    train_dataloader, _ = train_dataset.train_dataloader()

    print(f"Loaded training datapipe of length {len(train_dataloader)}")

    area = torch.abs(torch.cos(torch.linspace(-90, 90, steps=cfg.img_size[0]) * np.pi / 180))
    area /= torch.mean(area)
    area = area.unsqueeze(1)

    mean, mean_sqr = 0, 0
    for i, data in enumerate(train_dataloader):
        invar = data[0]  # [b, N, h, w]
        outvar = data[1]  # [b, N, h, w]
        diff = outvar - invar
        weighted_diff = area * diff
        weighted_diff_sqr = torch.square(weighted_diff)

        mean += torch.mean(weighted_diff, dim=(2, 3)) / len(train_dataloader)
        mean_sqr += torch.mean(weighted_diff_sqr, dim=(2, 3)) / len(train_dataloader)
        if (i+1) % 100 == 0:
            print(f"Number of iterations {i+1}/{len(train_dataloader)}")

    variance = mean_sqr - mean**2  # [1,num_channel, 1,1]
    std = torch.sqrt(variance)
    np.save("time_diff_std.npy", std.numpy())
    np.save("time_diff_mean.npy", mean.numpy())

    print(f"saving time_diff_std.npy and time_diff_mean, shapes are {std.numpy().shape}, {mean.numpy().shape}")


if __name__ == "__main__":
    current_path = os.getcwd()
    sys.path.append(current_path)
    main()

