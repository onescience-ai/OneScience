from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


ROOT_DIR = Path(__file__).resolve().parent


def resolve_path(path_value):
    path = Path(path_value)
    return path if path.is_absolute() else ROOT_DIR / path


def crop_latitude_band(data):
    if data.shape[-2] == 170:
        return data[..., :-10, :]
    if data.shape[-2] == 180:
        return data[..., 10:-10, :]
    return data


def load_stat_map(dataset_cfg, data_type):
    img_size = tuple(dataset_cfg.img_size)
    if data_type.endswith("Sin") or data_type.endswith("Cos"):
        return np.zeros(img_size, dtype=np.float32), np.ones(img_size, dtype=np.float32)

    stats_dir = resolve_path(dataset_cfg.stats_dir)
    mean_path = stats_dir / f"{data_type}_means.npy"
    std_path = stats_dir / f"{data_type}_stds.npy"

    if not mean_path.exists() or not std_path.exists():
        return np.zeros(img_size, dtype=np.float32), np.ones(img_size, dtype=np.float32)

    means = np.load(mean_path).astype(np.float32)
    stds = np.load(std_path).astype(np.float32)
    stds = np.where(stds == 0, 1.0, stds)
    return means, stds


def load_repeated_stats(dataset_cfg, data_types, repeat_steps):
    means = []
    stds = []
    specs = []
    for data_type in data_types:
        mean_map, std_map = load_stat_map(dataset_cfg, data_type)
        for lead_step in range(repeat_steps):
            means.append(mean_map)
            stds.append(std_map)
            specs.append(f"{data_type}@t+{lead_step + 1}")
    return np.stack(means, axis=0), np.stack(stds, axis=0), specs


def build_channel_sequence(base_types, repeat_steps):
    return [data_type for _ in range(repeat_steps) for data_type in base_types]


def get_input_channels(dataset_cfg):
    if "channels" in dataset_cfg:
        return list(dataset_cfg.channels)
    return build_channel_sequence(list(dataset_cfg.input_types), int(dataset_cfg.history_steps))


def get_output_channels(dataset_cfg):
    if "output_channels" in dataset_cfg:
        return list(dataset_cfg.output_channels)
    return build_channel_sequence(list(dataset_cfg.output_types), int(dataset_cfg.forecast_steps))


def get_output_specs(dataset_cfg):
    if "output_channels" not in dataset_cfg:
        _, _, specs = load_repeated_stats(
            dataset_cfg,
            list(dataset_cfg.output_types),
            int(dataset_cfg.forecast_steps),
        )
        return specs

    base_types = list(dataset_cfg.output_types)
    specs = []
    for index, data_type in enumerate(dataset_cfg.output_channels):
        lead_step = index // len(base_types) + 1 if len(base_types) > 0 else index + 1
        specs.append(f"{data_type}@t+{lead_step}")
    return specs


def load_or_create_mask(dataset_cfg):
    mask_path = resolve_path(dataset_cfg.static_dir) / "ocean_mask.npy"
    if mask_path.exists():
        return np.load(mask_path).astype(np.float32)

    source_candidates = [
        data_type for data_type in list(dataset_cfg.output_types) + list(dataset_cfg.input_types)
        if data_type.startswith("Wave") or data_type.startswith("Ocean")
    ]
    source_type = source_candidates[0] if source_candidates else dataset_cfg.input_types[0]
    for year in dataset_cfg.train_time + dataset_cfg.val_time + dataset_cfg.test_time:
        source_file = resolve_path(dataset_cfg.ocean_data_dir) / source_type / f"{year}.h5"
        if not source_file.exists():
            continue
        with h5py.File(source_file, "r") as h5_file:
            sample = h5_file["fields"][0:1]
        sample = crop_latitude_band(sample)[0]
        mask = np.where(np.isnan(sample), 0.0, 1.0).astype(np.float32)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(mask_path, mask)
        return mask

    mask = np.ones(tuple(dataset_cfg.img_size), dtype=np.float32)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(mask_path, mask)
    return mask


class OceanDatapipe:
    def __init__(self, dataset_cfg, dataloader_cfg, used_years, distributed=False, batch_size=None):
        self.dataset_cfg = dataset_cfg
        self.dataloader_cfg = dataloader_cfg
        self.used_years = used_years
        self.distributed = distributed
        self.batch_size = batch_size or dataloader_cfg.batch_size

    def get_dataloader(self, mode):
        dataset = OceanDataset(self.dataset_cfg, self.used_years)
        is_train = mode == "train"
        sampler = DistributedSampler(dataset, shuffle=is_train) if self.distributed else None

        dataloader_kwargs = {
            "dataset": dataset,
            "batch_size": self.batch_size,
            "num_workers": self.dataloader_cfg.num_workers,
            "pin_memory": self.dataloader_cfg.pin_memory,
            "shuffle": is_train and not self.distributed,
            "sampler": sampler,
            "drop_last": is_train and self.dataloader_cfg.drop_last,
        }
        if self.dataloader_cfg.num_workers > 0:
            dataloader_kwargs["persistent_workers"] = self.dataloader_cfg.persistent_workers
            dataloader_kwargs["prefetch_factor"] = self.dataloader_cfg.prefetch_factor

        return DataLoader(**dataloader_kwargs), sampler


class OceanDataset(Dataset):
    def __init__(self, dataset_cfg, used_years):
        self.dataset_cfg = dataset_cfg
        self.used_years = used_years
        self.input_types = list(dataset_cfg.input_types)
        self.output_types = list(dataset_cfg.output_types)
        self.history_steps = int(dataset_cfg.history_steps)
        self.forecast_steps = int(dataset_cfg.forecast_steps)
        self.input_channels = get_input_channels(dataset_cfg)
        self.output_channels = get_output_channels(dataset_cfg)
        self.normalize = dataset_cfg.normalization == "zscore"

        self.stats_cache = {}
        self.open_files = {}
        self.open_handles = {}
        self.sample_index = []

        self.in_channels = len(self.input_channels)
        self.out_channels = len(self.output_channels)

        self._build_index()

    def _build_index(self):
        for year in self.used_years:
            sample_file = self._resolve_data_file(self.input_types[0], year)
            if not sample_file.exists():
                raise FileNotFoundError(f"Missing dataset file: {sample_file}")

            with h5py.File(sample_file, "r") as h5_file:
                time_dim = h5_file["fields"].shape[0]

            usable = time_dim - self.history_steps - self.forecast_steps + 1
            if usable <= 0:
                raise ValueError(f"Year {year} does not have enough samples for the configured steps.")

            for start_idx in range(usable):
                self.sample_index.append((year, start_idx + self.history_steps))

    def _resolve_data_file(self, data_type, year):
        store_type = data_type[:-9] if data_type.endswith("_Forecast") else data_type

        if store_type.startswith("Wave") or store_type.startswith("Ocean"):
            return resolve_path(self.dataset_cfg.ocean_data_dir) / store_type / f"{year}.h5"
        if store_type.startswith("Wind"):
            return resolve_path(self.dataset_cfg.wind_data_dir) / store_type / f"{year}.h5"
        raise ValueError(f"Unsupported data type: {data_type}")

    def _get_dataset(self, year, data_type):
        store_type = data_type[:-9] if data_type.endswith("_Forecast") else data_type
        if year not in self.open_files:
            self.open_files[year] = {}
            self.open_handles[year] = {}
        if store_type not in self.open_files[year]:
            file_path = self._resolve_data_file(store_type, year)
            h5_file = h5py.File(file_path, "r")
            self.open_handles[year][store_type] = h5_file
            self.open_files[year][store_type] = h5_file["fields"]
        return self.open_files[year][store_type], store_type

    def _normalize_field(self, data, data_type):
        if data_type not in self.stats_cache:
            self.stats_cache[data_type] = load_stat_map(self.dataset_cfg, data_type)
        means, stds = self.stats_cache[data_type]

        if data_type.endswith("Sin") or data_type.endswith("Cos"):
            return np.nan_to_num(data, nan=0.0).astype(np.float32)

        data = np.where(np.isnan(data), means, data).astype(np.float32)
        if self.normalize:
            data = (data - means) / stds
        return data

    def _stack_inputs(self, year, start_idx):
        fields = []
        for channel_index, data_type in enumerate(self.input_channels):
            dataset, store_type = self._get_dataset(year, data_type)
            if data_type.endswith("_Forecast"):
                frame_idx = start_idx + channel_index // max(len(self.input_types), 1)
            else:
                frame_idx = start_idx - self.history_steps + channel_index // max(len(self.input_types), 1)
            field = crop_latitude_band(dataset[frame_idx:frame_idx + 1])[0]
            fields.append(self._normalize_field(field, store_type))
        return np.stack(fields, axis=0)

    def _stack_targets(self, year, start_idx):
        fields = []
        for channel_index, data_type in enumerate(self.output_channels):
            dataset, store_type = self._get_dataset(year, data_type)
            frame_idx = start_idx + channel_index // max(len(self.output_types), 1)
            field = crop_latitude_band(dataset[frame_idx:frame_idx + 1])[0]
            fields.append(self._normalize_field(field, store_type))
        return np.stack(fields, axis=0)

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        year, start_idx = self.sample_index[idx]
        inputs = self._stack_inputs(year, start_idx)
        targets = self._stack_targets(year, start_idx)
        return torch.from_numpy(inputs), torch.from_numpy(targets)
