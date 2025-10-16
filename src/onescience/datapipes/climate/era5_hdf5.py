import os
import glob
import json
import h5py
import pytz
import numpy as np
import torch

from datetime import datetime, timedelta
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from onescience.datapipes.datapipe import Datapipe
from onescience.datapipes.climate.utils.invariant import latlon_grid
from onescience.datapipes.climate.utils.zenith_angle import cos_zenith_angle


class ERA5HDF5Datapipe(Datapipe):
    def __init__(self, params, distributed, output_steps=1, input_steps=1):
        self.params = params
        self.distributed = distributed
        self.output_steps = output_steps
        self.input_steps = input_steps

    def train_dataloader(self):
        data = ERA5Dataset(params=self.params, mode='train', output_steps=self.output_steps, input_steps=self.input_steps)
        sampler = DistributedSampler(data, shuffle=True) if self.distributed else None
        data_loader = DataLoader(data,
                                 batch_size=self.params.batch_size,
                                 drop_last=True if self.distributed else False,
                                 num_workers=self.params.num_workers,
                                 pin_memory=True,
                                 shuffle=False,
                                 sampler=sampler)
        return data_loader, sampler

    def val_dataloader(self):
        data = ERA5Dataset(params=self.params, mode='val', output_steps=self.output_steps, input_steps=self.input_steps)
        sampler = DistributedSampler(data, shuffle=False) if self.distributed else None
        data_loader = DataLoader(data,
                                 batch_size=self.params.batch_size,
                                 drop_last=True if self.distributed else False,
                                 num_workers=self.params.num_workers,
                                 pin_memory=True,
                                 shuffle=False,
                                 sampler=sampler)
        return data_loader, sampler

    def test_dataloader(self):
        data = ERA5Dataset(params=self.params, mode='test', output_steps=self.output_steps, input_steps=self.input_steps)
        data_loader = DataLoader(data,
                                 batch_size=self.params.batch_size,
                                 drop_last=False,
                                 num_workers=self.params.num_workers,
                                 pin_memory=True,
                                 shuffle=False)
        return data_loader
    
    
class ERA5Dataset(Dataset):
    def __init__(self, params, mode='train', output_steps=1, input_steps=1, patch_size=[1, 1]):
        self.params = params
        self.data_dir = params.data_dir
        self.mode = mode
        self.output_steps = output_steps
        self.input_steps = input_steps
        self.patch_size = patch_size
        self.dt = params.time_res

        self.metadata = None
        self.years = []
        self.variables = []
        self.channel_indices = []
        self.mu = None
        self.sd = None
        self.selected_years = []
        self.files = {}
        self.samples_per_year = 0
        self.total_samples = 0
        self.img_shape = None
        self.latlon_torch = None

        self._init_paths()
        self._init_normalization()
        self._init_split()
        self._init_files()
        self._init_latlon()
        self._init_shape()

    def _init_paths(self):
        meta_path = os.path.join(self.data_dir, 'metadata.json')
        with open(meta_path, "r") as f:
            self.metadata = json.load(f)
        self.years = list(map(int, self.metadata["years"]))
        self.variables = self.metadata["variables"]

        # 检查 channels 是否都在 metadata.variables 中
        missing = [ch for ch in self.params.channels if ch not in self.variables]
        if missing:
            raise ValueError(f"❌ Missing required variables in metadata: {missing}")

    def _init_normalization(self):
        self.channel_indices = [self.variables.index(v) for v in self.params.channels]
        mu = np.load(os.path.join(self.params.stats_dir, "global_means.npy"))  # shape: [1, M, 1, 1]
        std = np.load(os.path.join(self.params.stats_dir, "global_stds.npy"))
        self.mu = mu[:, self.channel_indices, :, :]
        self.sd = std[:, self.channel_indices, :, :]

    def _init_split(self):
        y = sorted(self.years)
        if self.params.train_ratio + self.params.val_ratio + self.params.test_ratio == 1:
            n_train = int(len(y) * self.params.train_ratio)
            n_val = int(len(y) * self.params.val_ratio)
            year_splits = {
                "train": y[:n_train],
                "val": y[n_train:n_train + n_val],
                "test": y[n_train + n_val:]
            }
        elif self.params.train_ratio + self.params.val_ratio + self.params.test_ratio == len(y):
            n_train =  self.params.train_ratio
            n_val = self.params.val_ratio
            year_splits = {
                "train": y[:n_train],
                "val": y[n_train:n_train + n_val],
                "test": y[n_train + n_val:]
            }
        else:
            print('\n\n')
            print('-' * 30)
            print('Train/Val/Test settings must use ratio or digital numbers')
            print('If using ratio, please ensure the sum of all ratios equal to 1')
            print(f'If using digital number, please ensure the sum of number equal to total years {len(y)}')
            print(f'❌❌ Now settings are {self.params.train_ratio}-{self.params.val_ratio}-{self.params.test_ratio}, please check.')
            print('-' * 30)
            print('\n\n')
            exit()

        self.selected_years = year_splits[self.mode]
        

    def _init_files(self):
        for year in self.selected_years:
            path = os.path.join(self.data_dir, 'data', str(year))
            files = sorted(glob.glob(os.path.join(path, "*.h5")))
            self.files[year] = files
        self.samples_per_year = len(files) - self.output_steps - (self.input_steps - 1)
        self.total_samples = len(self.selected_years) * self.samples_per_year
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print('\n\n')
            print('-' * 50)
            print(f"📂 Mode: {self.mode}, used: {self.selected_years} years")
            print(f'📂 each years contains {self.samples_per_year} (Each year contains {len(files)}, input {self.input_steps}, output {self.output_steps})')
            print(f'📂 whole dataset contains {len(self.variables)} variables, this model use {len(self.channel_indices)} variables.')
            print(f'📂 {len(self.selected_years)} years * {self.samples_per_year} samples = Total {len(self.selected_years) * self.samples_per_year} usable samples.')
            print('-' * 50, '\n')

    def _init_latlon(self):
        latlon = latlon_grid(bounds=((90, -90), (0, 360)), shape=self.params.img_size[-2:])
        self.latlon_torch = torch.tensor(np.stack(latlon, axis=0), dtype=torch.float32)

    def _init_shape(self):
        sample_file = self.files[self.selected_years[0]][0]
        with h5py.File(sample_file, "r") as f:
            shape = f["fields"].shape  # [N, H, W]
            self.img_shape = [s - s % self.patch_size[i] for i, s in enumerate(shape[-2:])]

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        year_idx = idx // self.samples_per_year
        step_idx = idx % self.samples_per_year
        year = self.selected_years[year_idx]
        files = self.files[year]
        file_indices = range(step_idx, step_idx + self.input_steps + self.output_steps)

        data_list = []
        for i in file_indices:
            with h5py.File(files[i], "r") as f:
                data = f["fields"][:]  # [N, H, W]
                data = data[self.channel_indices]
                data_list.append(data)

        data = np.stack(data_list, axis=0)  # [T, N, H, W]
        invar = torch.as_tensor(data[:self.input_steps])
        outvar = torch.as_tensor(data[self.input_steps:])
        h, w = self.img_shape
        invar = invar[:, :, :h, :w]
        outvar = outvar[:, :, :h, :w]
        invar = (invar - self.mu) / self.sd
        outvar = (outvar - self.mu) / self.sd

        start_time = datetime(year, 1, 1, tzinfo=pytz.utc)
        timestamps = np.array([(start_time + timedelta(hours=(step_idx + t) * self.dt)).timestamp()
                               for t in range(self.output_steps)])
        timestamps = torch.from_numpy(timestamps)
        cos_zenith = cos_zenith_angle(timestamps, latlon=self.latlon_torch).float()

        return invar.squeeze(0), outvar.squeeze(0), cos_zenith, step_idx