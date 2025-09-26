import glob

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from onescience.datapipes.datapipe import Datapipe


class FuXiHDF5Datapipe(Datapipe):
    def __init__(self, params, distributed, mode, num_steps, input_steps=1):
        self.params = params
        self.distributed = distributed
        self.num_steps = num_steps
        self.input_steps = input_steps
        self.mode = mode

    def train_dataloader(self):
        data = ERA5Dataset(
            params=self.params,
            mode=self.mode,
            pred_paths=f"{self.params.train_data_dir}/{self.mode}",
            gt_paths=self.params.train_data_dir,
            num_steps=self.num_steps,
            input_steps=self.input_steps,
        )
        sampler = DistributedSampler(
            data, shuffle=True) if self.distributed else None
        data_loader = DataLoader(
            data,
            batch_size=self.params.batch_size,
            drop_last=True if self.distributed else False,
            num_workers=self.params.num_workers,
            pin_memory=True,
            shuffle=False,
            sampler=sampler,
        )
        return data_loader, sampler

    def val_dataloader(self):
        data = ERA5Dataset(
            params=self.params,
            mode=self.mode,
            pred_paths=f"{self.params.val_data_dir}/{self.mode}",
            gt_paths=self.params.val_data_dir,
            num_steps=self.num_steps,
            input_steps=self.input_steps,
        )
        sampler = DistributedSampler(
            data, shuffle=False) if self.distributed else None
        data_loader = DataLoader(
            data,
            batch_size=self.params.batch_size,
            drop_last=True if self.distributed else False,
            num_workers=self.params.num_workers,
            pin_memory=True,
            shuffle=False,
            sampler=sampler,
        )
        return data_loader, sampler

    def test_dataloader(self):
        data = ERA5Dataset(
            params=self.params,
            mode=self.mode,
            pred_paths=f"{self.params.test_data_dir}/{self.mode}",
            gt_paths=self.params.test_data_dir,
            num_steps=self.num_steps,
            input_steps=self.input_steps,
        )
        data_loader = DataLoader(
            data,
            batch_size=self.params.batch_size,
            drop_last=False,
            num_workers=self.params.num_workers,
            pin_memory=True,
            shuffle=False,
        )
        return data_loader


class ERA5Dataset(Dataset):
    def __init__(self, params, mode, pred_paths, gt_paths, num_steps=2, input_steps=1):
        self.params = params
        self.pred_paths = sorted(
            glob.glob(f"{pred_paths}/*.h5"))  # short_year.h5
        self.gt_paths = sorted(
            glob.glob(f"{gt_paths}/*.h5"))  # original year.h5
        self.num_steps = num_steps
        self.input_steps = input_steps
        self.mode = mode

        self.mu = np.load(
            f"{self.params.stats_dir}/global_means.npy")
        self.sd = np.load(
            f"{self.params.stats_dir}/global_stds.npy")

        self._load_metadata()

    def _load_metadata(self):
        with h5py.File(self.gt_paths[0], "r") as f:
            self.img_shape = f["fields"].shape[2:]
            self.channels = [
                i for i in range(f["fields"].shape[1])]

        with h5py.File(self.pred_paths[0], "r") as f:
            self.samples_per_year = (
                f["fields"].shape[0] - self.num_steps -
                (self.input_steps - 1)
            )
        self.n_years = len(self.pred_paths)
        self.total_length = self.n_years * self.samples_per_year

        self.pred_files = None
        self.gt_files = None

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        if self.pred_files is None:
            self.pred_files = [
                h5py.File(p, "r") for p in self.pred_paths]
            self.gt_files = [
                h5py.File(p, "r") for p in self.gt_paths]
        year_idx = idx // self.samples_per_year
        step_idx = idx % self.samples_per_year
        step_idx = 0

        invar = self.pred_files[year_idx]["fields"][
            step_idx: step_idx + self.input_steps
        ]  # [C, H, W]
        invar = torch.as_tensor(invar)

        if self.mode == "medium":
            out_idx = (
                step_idx
                + (self.params.medium_num_steps -
                   self.params.short_num_steps)
                + (self.input_steps * 2 - 1)
            )
        if self.mode == "long":
            out_idx = (
                step_idx
                + (self.params.long_num_steps -
                   self.params.medium_num_steps)
                + (self.input_steps * 2 - 1)
            )

        outvar = self.gt_files[year_idx]["fields"][
            out_idx: out_idx + self.num_steps
        ]  # [T, C, H, W]
        outvar = torch.as_tensor(outvar)

        h, w = self.img_shape
        invar = invar[:, :h, :w]
        outvar = outvar[:, :, :h, :w]

        invar = (invar - self.mu) / self.sd
        outvar = (outvar - self.mu) / self.sd

        return invar, outvar
