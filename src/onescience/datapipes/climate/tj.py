import bisect
import glob
import json
import os

import numpy as np
import pytz
import torch
import xarray as xr

from datetime import datetime, timedelta
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from onescience.datapipes.climate.utils.invariant import latlon_grid
from onescience.datapipes.climate.utils.zenith_angle import cos_zenith_angle
from onescience.datapipes.core import BaseDataset
from onescience.datapipes.datapipe import Datapipe


def _normalize_variables(values):
    normalized = []
    for value in values:
        if isinstance(value, bytes):
            normalized.append(value.decode())
        else:
            normalized.append(str(value))
    return normalized


class TJDatapipe(Datapipe):
    def __init__(self, params, distributed, output_steps=1, input_steps=1, normalize=True):
        self.params = params
        self.dataset = params.dataset
        self.distributed = distributed
        self.output_steps = output_steps
        self.input_steps = input_steps
        self.normalize = normalize

    def _build_dataloader(self, mode):
        data = TJDataset(
            dataset=self.dataset,
            mode=mode,
            output_steps=self.output_steps,
            input_steps=self.input_steps,
            normalize=self.normalize,
        )
        is_train = mode == "train"
        sampler = DistributedSampler(data, shuffle=is_train) if self.distributed else None
        data_loader = DataLoader(
            data,
            batch_size=self.params.dataloader.batch_size,
            drop_last=True if self.distributed else False,
            num_workers=self.params.dataloader.num_workers,
            pin_memory=getattr(self.params.dataloader, "pin_memory", True),
            shuffle=False if self.distributed else is_train,
            sampler=sampler,
        )
        return data_loader, sampler

    def train_dataloader(self):
        return self._build_dataloader("train")

    def val_dataloader(self):
        return self._build_dataloader("val")

    def test_dataloader(self):
        data = TJDataset(
            dataset=self.dataset,
            mode="test",
            output_steps=self.output_steps,
            input_steps=self.input_steps,
            normalize=self.normalize,
        )
        data_loader = DataLoader(
            data,
            batch_size=1,
            drop_last=True if self.distributed else False,
            num_workers=getattr(self.params.dataloader, "num_workers", 0),
            pin_memory=getattr(self.params.dataloader, "pin_memory", True),
            shuffle=False,
        )
        return data_loader


class TJDataset(BaseDataset):
    def __init__(self, dataset, mode="train", output_steps=1, input_steps=1, normalize=True, patch_size=[1, 1]):
        self.params = dataset
        self.data_dir = self.params.data_dir
        self.mode = mode
        self.output_steps = output_steps
        self.input_steps = input_steps
        self.patch_size = patch_size
        self.dt = self.params.time_res
        self.normalize = normalize

        self.metadata = None
        self.years = []
        self.variables = []
        self.channel_indices = []
        self.mu = None
        self.sd = None
        self.selected_years = []
        self.files = {}
        self.samples_per_year = 0
        self.sample_counts = {}
        self.year_offsets = []
        self.total_samples = 0
        self.img_shape = None
        self.latlon_torch = None
        self.channel_dim = "channel"
        self.lat_dim = "latitude"
        self.lon_dim = "longitude"

        self._init_paths()
        self._init_normalization()
        self._init_split()
        self._init_files()
        self._init_latlon()
        self._init_shape()
        super().__init__(self.params)

    def _init_paths(self):
        meta_path = os.path.join(self.data_dir, "metadata.json")
        with open(meta_path, "r") as f:
            self.metadata = json.load(f)
        self.years = list(map(int, self.metadata["years"]))
        self.variables = self.metadata["variables"]

        missing = [ch for ch in self.params.channels if ch not in self.variables]
        if missing:
            raise ValueError(f"❌ Missing required variables in metadata: {missing}")

    def _init_normalization(self):
        self.channel_indices = [self.variables.index(v) for v in self.params.channels]
        mu = np.load(os.path.join(self.params.stats_dir, "global_means.npy"))
        std = np.load(os.path.join(self.params.stats_dir, "global_stds.npy"))
        self.mu = mu[:, self.channel_indices, :, :]
        self.sd = std[:, self.channel_indices, :, :]

    def _init_split(self):
        available_years = set(self.years)
        split_attr = f"{self.mode}_years"
        legacy_attr = f"{self.mode}_ratio"

        selected_years = getattr(self.params, split_attr, None)
        if selected_years is None:
            selected_years = getattr(self.params, legacy_attr, None)

        if not isinstance(selected_years, (list, tuple, set)):
            raise ValueError(
                f"❌ TJ {self.mode} split must be an explicit year list/tuple/set. "
                f"Got {legacy_attr}={selected_years}. "
                f"Available years: {sorted(self.years)}"
            )

        self.selected_years = list(dict.fromkeys(int(year) for year in selected_years))
        if not self.selected_years:
            raise ValueError(
                f"❌ TJ {self.mode} split is empty. "
                f"Please provide explicit years from {sorted(self.years)}."
            )

        missing_years = sorted(year for year in self.selected_years if year not in available_years)
        if missing_years:
            raise ValueError(
                f"❌ Years not found in dataset for mode={self.mode}: {missing_years}. "
                f"Available years: {sorted(self.years)}"
            )

    def _init_files(self):
        total_samples = 0
        for year in self.selected_years:
            path = os.path.join(self.data_dir, "data", str(year))
            files = sorted(glob.glob(os.path.join(path, "*.nc")))
            if not files:
                raise ValueError(f"❌ No TJ files found for year {year} under {path}")

            year_samples = len(files) - self.output_steps - self.input_steps + 1
            if year_samples <= 0:
                raise ValueError(
                    f"❌ Year {year} does not contain enough samples. "
                    f"Found {len(files)} files, but input_steps={self.input_steps}, "
                    f"output_steps={self.output_steps}."
                )

            self.files[year] = files
            self.sample_counts[year] = year_samples
            self.year_offsets.append(total_samples)
            total_samples += year_samples

        unique_sample_counts = set(self.sample_counts.values())
        self.samples_per_year = unique_sample_counts.pop() if len(unique_sample_counts) == 1 else None
        self.total_samples = total_samples
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print("\n")
            print("-" * 50)
            print(f"📂 Mode: {self.mode}, used: {self.selected_years} years")
            if self.samples_per_year is not None:
                print(
                    f"📂 each selected year contains {self.samples_per_year} usable samples "
                    f"(input {self.input_steps}, output {self.output_steps})"
                )
            else:
                print("📂 usable samples vary by year:")
                for year in self.selected_years:
                    print(
                        f"   - {year}: {self.sample_counts[year]} usable samples "
                        f"from {len(self.files[year])} files"
                    )
            print(f"📂 whole dataset contains {len(self.variables)} variables, this model use {len(self.channel_indices)} variables.")
            print(f"📂 total usable samples: {self.total_samples}")
            print("-" * 50, "\n")

    def _init_latlon(self):
        latlon = latlon_grid(bounds=((90, -90), (0, 360)), shape=self.params.img_size[-2:])
        self.latlon_torch = torch.tensor(np.stack(latlon, axis=0), dtype=torch.float32)

    def _infer_spatial_dims(self, fields):
        dims = list(fields.dims)
        for candidate in ("channel", "variable", "variables"):
            if candidate in dims:
                self.channel_dim = candidate
                break
        else:
            self.channel_dim = dims[0]

        for candidate in ("latitude", "lat"):
            if candidate in dims:
                self.lat_dim = candidate
                break
        else:
            self.lat_dim = dims[-2]

        for candidate in ("longitude", "lon"):
            if candidate in dims:
                self.lon_dim = candidate
                break
        else:
            self.lon_dim = dims[-1]

    def _init_shape(self):
        sample_file = self.files[self.selected_years[0]][0]
        with xr.open_dataset(sample_file) as ds:
            if "fields" not in ds:
                raise ValueError(f"❌ Expected variable 'fields' in {sample_file}")

            fields = ds["fields"]
            self._infer_spatial_dims(fields)
            fields = fields.transpose(self.channel_dim, self.lat_dim, self.lon_dim)

            if self.channel_dim in fields.coords:
                file_variables = _normalize_variables(fields.coords[self.channel_dim].values.tolist())
                missing = [ch for ch in self.params.channels if ch not in file_variables]
                if missing:
                    raise ValueError(f"❌ Variables not found in TJ file {sample_file}: {missing}")
                self.channel_indices = [file_variables.index(v) for v in self.params.channels]

            shape = fields.shape
            self.img_shape = [s - s % self.patch_size[i] for i, s in enumerate(shape[-2:])]

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        year_idx = bisect.bisect_right(self.year_offsets, idx) - 1
        step_idx = idx - self.year_offsets[year_idx]
        year = self.selected_years[year_idx]
        files = self.files[year]
        file_indices = range(step_idx, step_idx + self.input_steps + self.output_steps)

        data_list = []
        time_index = []
        for i in file_indices:
            with xr.open_dataset(files[i]) as ds:
                fields = ds["fields"].transpose(self.channel_dim, self.lat_dim, self.lon_dim)
                data = fields.isel({self.channel_dim: self.channel_indices}).values
                data_list.append(data)
                time_index.append(os.path.splitext(os.path.basename(files[i]))[0])

        data = np.stack(data_list, axis=0)
        invar = torch.as_tensor(data[:self.input_steps])
        outvar = torch.as_tensor(data[self.input_steps:])
        h, w = self.img_shape
        invar = invar[:, :, :h, :w]
        outvar = outvar[:, :, :h, :w]
        if self.normalize:
            invar = (invar - self.mu) / self.sd
            outvar = (outvar - self.mu) / self.sd

        start_time = datetime(year, 1, 1, tzinfo=pytz.utc)
        timestamps = np.array(
            [
                (start_time + timedelta(hours=(step_idx + t) * self.dt)).timestamp()
                for t in range(self.output_steps)
            ]
        )
        timestamps = torch.from_numpy(timestamps)
        cos_zenith = cos_zenith_angle(timestamps, latlon=self.latlon_torch).float()

        return invar.squeeze(0), outvar.squeeze(0), cos_zenith, step_idx, time_index
