# This data_loader is adapted from data_loader_multifiles.py.
# The data is loading from different source, such as wave period, wave height, etc.
# You can add other data source by adding data in main_folder/train.

import glob
import os

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


def get_data_loader(params, distributed, mode):
    dataset = GetDataset(params, mode)
    if mode == "train":
        drop_last = True
        shuffle = True
        batch_size = int(params.batch_size)
        sampler = DistributedSampler(
            dataset, shuffle=shuffle) if distributed else None
    elif mode == "valid":
        drop_last = False
        shuffle = False
        batch_size = int(params.batch_size)
        sampler = DistributedSampler(
            dataset, shuffle=shuffle) if distributed else None
    else:
        drop_last = False
        sampler = None
        batch_size = int(params.batch_size)

    if mode == "test":
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=4,
            shuffle=False,
            sampler=sampler,
            drop_last=drop_last,
            pin_memory=torch.cuda.is_available(),
        )
        return dataloader, dataset
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=params.num_data_workers,
            shuffle=False,
            sampler=sampler,
            drop_last=drop_last,
            pin_memory=torch.cuda.is_available(),
        )
        return dataloader, dataset, sampler


class GetDataset(Dataset):
    def __init__(self, params, mode):
        self.params = params
        self.mode = mode
        self.in_chans = params.in_channels
        self.out_chans = params.out_channels
        self.input_types = params.input_types
        self.output_types = params.output_types
        self.n_history = params.n_history
        self.dt = params.dt
        if self.output_types[0].startswith("Wave"):
            self.n_samples_per_year = 2920
        else:
            self.n_samples_per_year = 365
        self.img_shape_x = params.image_width
        self.img_shape_y = params.image_height

        self._get_files_stats()
        self.add_noise = True if mode == "train" else False

        try:
            self.normalize = params.normalize
        except:
            self.normalize = True

    def _get_files_stats(self):
        self.file_path = glob.glob(
            f"{self.params.count_data_path}/*.h5")
        self.file_path = [os.path.basename(
            file) for file in self.file_path]
        self.file_path.sort()

        # select train-valid-test dataset according to 8-1-1
        total_files = len(self.file_path)
        mid_1_idx = int(0.8 * total_files)
        mid_2_idx = int(0.9 * total_files)
        if self.mode == "train":
            self.file_path = self.file_path[:mid_1_idx]
        elif self.mode == "valid":
            self.file_path = self.file_path[mid_1_idx:mid_2_idx]
        elif self.mode == "test":
            self.file_path = self.file_path[mid_2_idx:]
        else:
            raise ValueError(
                f"Invalid mode {self.mode}. Use 'train', 'valid', or 'test'."
            )
        if not self.file_path:
            raise RuntimeError(
                f"No .h5 files found in {self.params.wind_data_path}")
        self.n_years = len(self.file_path)
        self.files = [{} for _ in range(self.n_years)]
        self.n_samples_total = self.n_years * self.n_samples_per_year

    def _open_file(self, year_idx):

        self.files[year_idx] = {}
        for data_type in self.input_types:
            # For temporary replacement purposes
            if data_type.endswith("Forecast"):
                data_type = data_type[: -len("_Forecast")]
            if data_type.startswith("Wave"):
                file_path = os.path.join(
                    self.params.ocean_data_path, data_type, self.file_path[year_idx]
                )
            if data_type.startswith("Ocean"):
                file_path = os.path.join(
                    self.params.ocean_data_path, data_type, self.file_path[year_idx]
                )
            if data_type.startswith("Current"):
                file_path = os.path.join(
                    self.params.current_data_path, data_type, self.file_path[year_idx]
                )
            if data_type.startswith("Wind"):
                file_path = os.path.join(
                    self.params.wind_data_path, data_type, self.file_path[year_idx]
                )
            _files = h5py.File(file_path, "r")
            self.files[year_idx][data_type] = _files[list(_files.keys())[
                0]]

    def __len__(self):
        return self.n_samples_total

    def __getitem__(self, global_idx):

        year_idx = int(global_idx / self.n_samples_per_year)
        local_idx = int(global_idx %
                        self.n_samples_per_year)

        if not self.files[year_idx]:
            self._open_file(year_idx)
        if local_idx < self.n_history:
            local_idx = self.n_history
        if local_idx > self.n_samples_per_year - self.dt - 1:
            local_idx = self.n_samples_per_year - self.dt - 1

        inp_data_list = []
        for data_type in self.input_types:
            if data_type.endswith("Forecast"):
                data_type = data_type[: -len("_Forecast")]
                for step in range(0, self.dt):
                    start_idx = local_idx + step
                    end_idx = local_idx + step + 1
                    data = self.files[year_idx][data_type][start_idx:end_idx]
                    if data.shape[-2] == 170:
                        data = data[:, :-10]
                    if data.shape[-2] == 180:
                        data = data[:, 10:-10]
                    data_reshaped = reshape_fields(
                        local_idx,
                        start_idx,
                        end_idx,
                        "inp",
                        data,
                        data_type,
                        self.params,
                        self.normalize,
                        self.add_noise,
                    )
                    inp_data_list.append(data_reshaped)
            else:
                for step in range(self.n_history, 0, -1):
                    start_idx = local_idx - step
                    end_idx = local_idx - step + 1
                    data = self.files[year_idx][data_type][start_idx:end_idx]
                    if data.shape[-2] == 170:
                        data = data[:, :-10]
                    if data.shape[-2] == 180:
                        data = data[:, 10:-10]
                    data_reshaped = reshape_fields(
                        local_idx,
                        start_idx,
                        end_idx,
                        "inp",
                        data,
                        data_type,
                        self.params,
                        self.normalize,
                        self.add_noise,
                    )
                    inp_data_list.append(data_reshaped)
        inp_data = np.concatenate(inp_data_list, axis=0)

        tar_data_list = []
        for data_type in self.output_types:
            for step in range(0, self.dt):
                start_idx = local_idx + step
                end_idx = local_idx + step + 1
                data = self.files[year_idx][data_type][start_idx:end_idx]
                if data.shape[-2] == 170:
                    data = data[:, :-10]
                if data.shape[-2] == 180:
                    data = data[:, 10:-10]
                data_reshaped = reshape_fields(
                    local_idx, start_idx, end_idx, "tar", data, data_type, self.params
                )
                tar_data_list.append(data_reshaped)
        tar_data = np.concatenate(tar_data_list, axis=0)
        return torch.as_tensor(inp_data), torch.as_tensor(tar_data)


def reshape_fields(
    local,
    start,
    end,
    inp_or_tar,
    img,
    data_type,
    params,
    normalize=True,
    add_noise=False,
):

    if data_type.endswith("Sin") or data_type.endswith("Cos"):
        img = np.where(np.isnan(img), 0, img)
    else:
        means = np.load(
            f"{params.global_means_path}/{data_type}_means.npy")
        stds = np.load(
            f"{params.global_stds_path}/{data_type}_stds.npy")
        img = np.where(np.isnan(img), means, img)
        if normalize:
            if params.normalization == "minmax":
                raise Exception(
                    "minmax not supported. Use zscore")
            if params.normalization == "zscore":
                img = (img - means) / stds
    if add_noise:
        img = img + \
            np.random.normal(
                0, scale=params.noise_std, size=img.shape)

    return img
