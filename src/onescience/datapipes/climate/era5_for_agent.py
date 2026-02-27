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
from onescience.datapipes.core import BaseDataset


class ERA5Datapipe(Datapipe):
    def __init__(self, params, distributed=False):
        self.params = params
        self.distributed = distributed

    def dataloader(self):
        data = ERA5Dataset(params=self.params)
        sampler = DistributedSampler(data, shuffle=True) if self.distributed else None
        data_loader = DataLoader(data,
                                 batch_size=self.params['batch_size'],
                                 drop_last=True if self.distributed else False,
                                 num_workers=self.params['num_workers'],
                                 pin_memory=True,
                                 shuffle=False,
                                 sampler=sampler)
        return data_loader, sampler
    
    
class  ERA5Dataset(BaseDataset):
    def __init__(self, params):
        self.data_dir = params['data_dir']
        self.stats_dir = params['stats_dir']
        self.used_years = params['used_years']
        self.used_channels = params['used_channels']
        self.output_steps = params['output_steps']
        self.input_steps = params['input_steps']
        self.normalize = params['normalize']

        self.metadata = None
        self.years = []
        self.variables = []
        self.channel_indices = []
        self.mu = None
        self.sd = None
        self.files = {}
        self.samples_per_year = 0
        self.total_samples = 0
        self.img_shape = None
        self.latlon_torch = None

        self._init_paths()
        self._init_normalization()
        self._init_years()
        self._init_files()


    def _init_paths(self):
        meta_path = os.path.join(self.data_dir, 'metadata.json')
        with open(meta_path, "r") as f:
            self.metadata = json.load(f)
        self.years = list(map(int, self.metadata["years"]))
        self.variables = self.metadata["variables"]

        # 检查 channels 是否都在 metadata.variables 中
        missing = [ch for ch in self.used_channels if ch not in self.variables]
        if missing:
            raise ValueError(f"❌ Missing required variables in metadata: {missing}")


    def _init_normalization(self):
        self.channel_indices = [self.variables.index(v) for v in self.used_channels]
        mu = np.load(os.path.join(self.stats_dir, "global_means.npy"))  # shape: [1, M, 1, 1]
        std = np.load(os.path.join(self.stats_dir, "global_stds.npy"))
        self.mu = mu[:, self.channel_indices, :, :]
        self.sd = std[:, self.channel_indices, :, :]


    def _init_years(self):
        y = sorted(self.years)
        tips = False
        error = False

        tmp_used_years = set(self.used_years)
        if not tmp_used_years.issubset(set(self.years)):
            print('\n')
            print('-' * 50)
            print(f'❌ ❌ please ensure the years are exist in provided dataset.')
            print(f'We provided {len(y)} years data, which are {y}')
            print(f'❌ ❌ Now settings are train: {self.used_years}')
            print('-' * 50)
            exit()
        

    def _init_files(self):
        for year in self.used_years:
            path = os.path.join(self.data_dir, 'data', str(year))
            files = sorted(glob.glob(os.path.join(path, "*.h5")))
            self.files[year] = files
        self.samples_per_year = len(files) - self.output_steps - (self.input_steps - 1)
        self.total_samples = len(self.used_years) * self.samples_per_year
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print('\n')
            print('-' * 50)
            print(f"📂 Now using years: {self.used_years} ")
            print(f'📂 each years contains {self.samples_per_year} (Each year contains {len(files)}, input {self.input_steps}, output {self.output_steps})')
            print(f'📂 whole dataset contains {len(self.variables)} variables, this model use {len(self.channel_indices)} variables.')
            print(f'📂 {len(self.used_years)} years * {self.samples_per_year} samples = Total {len(self.used_years) * self.samples_per_year} usable samples.')
            print('-' * 50, '\n')


    def __len__(self):
        return self.total_samples


    def __getitem__(self, idx):
        year_idx = idx // self.samples_per_year
        step_idx = idx % self.samples_per_year
        year = self.used_years[year_idx]
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
        if self.normalize:
            invar = (invar - self.mu) / self.sd
            outvar = (outvar - self.mu) / self.sd

        return invar.squeeze(0), outvar.squeeze(0)