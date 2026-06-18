import glob
import json
import os

import h5py
import numpy as np
import pytz
import torch

from datetime import datetime, timedelta
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from onescience.datapipes.climate.utils.invariant import latlon_grid
from onescience.datapipes.climate.utils.zenith_angle import cos_zenith_angle


def _normalize_variable_names(values):
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


class CMEMSDatapipe:
    def __init__(
        self,
        dataset_dir,
        used_years,
        used_variables,
        distributed=False,
        input_steps=1,
        output_steps=1,
        normalize=True,
        batch_size=1,
        num_workers=4,
    ):
        self.dataset_dir = dataset_dir
        self.used_years = used_years
        self.used_variables = used_variables
        self.distributed = distributed
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.normalize = normalize
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_dataloader(self, mode):
        dataset = CMEMSDataset(
            dataset_dir=self.dataset_dir,
            used_years=self.used_years,
            used_variables=self.used_variables,
            input_steps=self.input_steps,
            output_steps=self.output_steps,
            normalize=self.normalize,
        )
        is_train = mode == "train"

        sampler = DistributedSampler(dataset, shuffle=is_train) if self.distributed else None

        return DataLoader(
            dataset,
            batch_size=1 if mode == "test" else self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=(is_train and not self.distributed),
            sampler=sampler,
            drop_last=self.distributed,
        ), sampler


class CMEMSDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        used_years,
        used_variables,
        mode="train",
        input_steps=1,
        output_steps=1,
        normalize=True,
    ):
        self.dataset_dir = dataset_dir
        self.used_years = used_years
        self.used_variables = used_variables
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.normalize = normalize

        self.metadata = {}

        self._init_avail_samples()
        self._init_normalized_files()
        self._init_latlon_grid()

    def _init_avail_samples(self):
        meta_path = os.path.join(self.dataset_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                self.metadata = json.load(f)

        h5_files = sorted(glob.glob(os.path.join(self.dataset_dir, "data", "*.h5")))
        if not h5_files:
            raise ValueError(
                f"❌ No CMEMS HDF5 files found under {os.path.join(self.dataset_dir, 'data')}."
            )

        available_years = [int(os.path.basename(f).replace(".h5", "")) for f in h5_files]
        missing_years = [year for year in self.used_years if year not in available_years]
        if missing_years:
            raise ValueError(f"❌ Years not found in dataset: {missing_years}")

        with h5py.File(h5_files[0], "r") as f:
            if "fields" not in f:
                raise ValueError(f"❌ Expected dataset 'fields' in {h5_files[0]}")

            ds = f["fields"]
            if ds.ndim != 4:
                raise ValueError(
                    f"❌ Expected 'fields' to have shape [T, C, H, W], got {ds.shape} in {h5_files[0]}"
                )

            self.T, self.C, self.H, self.W = ds.shape

            if "variables" in ds.attrs:
                all_variables = _normalize_variable_names(ds.attrs["variables"])
            elif "variables" in self.metadata:
                all_variables = [str(v) for v in self.metadata["variables"]]
            else:
                raise ValueError(
                    f"❌ Missing variable names in attrs['variables'] for {h5_files[0]} and metadata.json."
                )

            if "time_step" in ds.attrs:
                self.time_step = int(ds.attrs["time_step"])
            else:
                raise ValueError(f"❌ Missing attrs['time_step'] in {h5_files[0]}")

        missing_vars = [var for var in self.used_variables if var not in all_variables]
        if missing_vars:
            raise ValueError(f"❌ Variables not found in dataset: {missing_vars}")

        self.channel_indices = [all_variables.index(var) for var in self.used_variables]
        self.file_map = {
            year: os.path.join(self.dataset_dir, "data", f"{year}.h5")
            for year in self.used_years
        }
        self.samples_per_year = self.T - self.input_steps - self.output_steps + 1
        self.total_samples = len(self.used_years) * self.samples_per_year

    def _init_normalized_files(self):
        stats_dir = os.path.join(self.dataset_dir, "stats")
        mu = np.load(os.path.join(stats_dir, "global_means.npy"))
        std = np.load(os.path.join(stats_dir, "global_stds.npy"))
        self.mu = torch.as_tensor(mu[:, self.channel_indices, :, :], dtype=torch.float32)
        self.sd = torch.as_tensor(std[:, self.channel_indices, :, :], dtype=torch.float32)

    def _init_latlon_grid(self):
        latlon = latlon_grid(bounds=((90, -90), (0, 360)), shape=(self.H, self.W))
        self.latlon_torch = torch.tensor(np.stack(latlon, axis=0), dtype=torch.float32)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        year_idx = idx // self.samples_per_year
        step_idx = idx % self.samples_per_year
        year = self.used_years[year_idx]

        with h5py.File(self.file_map[year], "r") as f:
            frames = f["fields"][step_idx: step_idx + self.input_steps + self.output_steps]
            frames = frames[:, self.channel_indices, :, :]

        data = torch.as_tensor(frames, dtype=torch.float32)
        invar = data[:self.input_steps]
        outvar = data[self.input_steps:]

        invar = torch.where(torch.isnan(invar), self.mu, invar)
        outvar = torch.where(torch.isnan(outvar), self.mu, outvar)

        if self.normalize:
            invar = (invar - self.mu) / self.sd
            outvar = (outvar - self.mu) / self.sd

        start_time = datetime(year, 1, 1, tzinfo=pytz.utc)
        timestamps = np.array([
            (start_time + timedelta(hours=(step_idx + self.input_steps + t) * self.time_step)).timestamp()
            for t in range(self.output_steps)
        ])
        timestamps = torch.from_numpy(timestamps)
        cos_zenith = cos_zenith_angle(timestamps, latlon=self.latlon_torch).float()

        time_index = [
            (start_time + timedelta(hours=(step_idx + t) * self.time_step)).strftime("%Y%m%d%H")
            for t in range(self.input_steps + self.output_steps)
        ]

        return invar.squeeze(0), outvar.squeeze(0), cos_zenith, step_idx, time_index