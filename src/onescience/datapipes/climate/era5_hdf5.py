import h5py
import numpy as np
import torch
import glob
import pytz
from ..datapipe import Datapipe
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from onescience.datapipes.climate.utils.invariant import latlon_grid
from onescience.datapipes.climate.utils.zenith_angle import cos_zenith_angle
from datetime import datetime, timedelta


class ERA5HDF5Datapipe(Datapipe):
    def __init__(self, params, distributed, num_steps=1, input_steps=1):
        self.params = params
        self.distributed = distributed
        self.num_steps = num_steps
        self.input_steps = input_steps

    def train_dataloader(self):
        data = ERA5Dataset(params=self.params, data_paths=self.params.train_data_dir, num_steps=self.num_steps, input_steps=self.input_steps)
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
        data = ERA5Dataset(params=self.params, data_paths=self.params.val_data_dir, num_steps=self.num_steps, input_steps=self.input_steps)
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
        data = ERA5Dataset(params=self.params, data_paths=self.params.test_data_dir, num_steps=self.num_steps, input_steps=self.input_steps)
        data_loader = DataLoader(data,
                                 batch_size=self.params.batch_size,
                                 drop_last=False,
                                 num_workers=self.params.num_workers,
                                 pin_memory=True,
                                 shuffle=False)
        return data_loader


class ERA5Dataset(Dataset):
    def __init__(self, params, data_paths, num_steps=1, input_steps=1, patch_size=[1, 1]):
        self.params = params
        self.data_files = None
        self.data_dir = data_paths

        self.mu = np.load(f'{self.params.stats_dir}/global_means.npy')
        self.sd = np.load(f'{self.params.stats_dir}/global_stds.npy')
        self.patch_size = patch_size
        self.num_steps = num_steps
        self.input_steps = input_steps
        self.parse_dataset_files()

    def parse_dataset_files(self):
        self.data_paths = glob.glob(f'{self.data_dir}/*.h5')
        self.data_paths.sort()
        self.n_years = len(self.data_paths)
        with h5py.File(self.data_paths[0], "r") as f:
            data_samples_per_year = f["fields"].shape[0]
            self.img_shape = f["fields"].shape[2:]
            self.channels = [i for i in range(f["fields"].shape[1])]
            self.img_shape = [s - s % self.patch_size[i] for i, s in enumerate(self.img_shape)]
        for i in range(len(self.data_paths)):
            with h5py.File(self.data_paths[i], "r") as f:
                if data_samples_per_year > f["fields"].shape[0]:
                    data_samples_per_year = f["fields"].shape[0]
        self.num_samples_per_year = data_samples_per_year - self.num_steps - (self.input_steps - 1)
        self.total_length = self.n_years * self.num_samples_per_year
        self.start_year = int(self.data_paths[0][-7:-3])
        self.dt = self.params.time_res
        self.latlon_bounds = ((90, -90), (0, 360))
        latlon = latlon_grid(bounds=self.latlon_bounds, shape=self.params.img_size[-2:])
        self.latlon_torch = torch.tensor(np.stack(latlon, axis=0), dtype=torch.float32)

    def __getitem__(self, idx):
        if self.data_files is None:
            self.data_files = [h5py.File(path, "r") for path in self.data_paths]

        file_idx = idx // self.num_samples_per_year
        step_idx = idx % self.num_samples_per_year

        invar_data = self.data_files[file_idx]["fields"]
        invar = invar_data[step_idx: step_idx + self.input_steps]
        # shape is [1, N, H, W]

        outvar_data = self.data_files[file_idx]["fields"]
        outvar = outvar_data[step_idx + self.input_steps: step_idx + self.input_steps + self.num_steps]
        # shape is [self.num_steps, N, H, W]

        invar = torch.as_tensor(invar)
        outvar = torch.as_tensor(outvar)
        h, w = self.img_shape
        invar = invar[:, :, :h, :w]
        outvar = outvar[:, :, :h, :w]

        invar = (invar - self.mu) / self.sd
        outvar = (outvar - self.mu) / self.sd

        start_time = datetime(self.start_year + file_idx, 1, 1, tzinfo=pytz.utc)
        timestamps = np.array([(start_time + timedelta(hours=(step_idx + 1 + t) * self.dt)).timestamp() for t in range(self.num_steps)])
        timestamps = torch.from_numpy(timestamps)
        cos_zenith = cos_zenith_angle(timestamps, latlon=self.latlon_torch).float()

        return invar.squeeze(0), outvar.squeeze(0), cos_zenith, step_idx

    def __len__(self):
        return self.total_length  # // self.batch_size