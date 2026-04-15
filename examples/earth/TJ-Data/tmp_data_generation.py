import json
import os
import sys

import numpy as np


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")

for path in (PROJECT_ROOT, SRC_ROOT):
    if path not in sys.path:
        sys.path.append(path)

from onescience.utils.YParams import YParams

try:
    from netCDF4 import Dataset as NetCDFDataset
except ImportError:  # pragma: no cover
    NetCDFDataset = None


def _resolve_path(base_dir, path_value):
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(base_dir, path_value))


def _collect_years(dataset_cfg):
    years = []
    for attr in ("train_years", "val_years", "test_years", "used_years"):
        values = getattr(dataset_cfg, attr, None)
        if values is not None:
            years.extend(int(year) for year in values)
    return sorted(set(years))


def _chunk_size(height, width):
    return (1, min(height, 180), min(width, 360))


def _timestamp_strings(year, num_steps, time_res):
    base = np.datetime64(f"{year}-01-01T00")
    times = base + np.arange(num_steps) * np.timedelta64(int(time_res), "h")
    return [
        np.datetime_as_string(value.astype("datetime64[h]"), unit="h").replace("-", "").replace("T", "")[:10]
        for value in times
    ]


def _create_sparse_zero_template(output_path, cfg):
    if NetCDFDataset is None:
        raise ImportError(
            "netCDF4 is required for sparse zero mock generation. "
            "Please install netCDF4 in your runtime environment."
        )

    channels = list(cfg.channels)
    height, width = map(int, cfg.img_size)

    with NetCDFDataset(output_path, "w", format="NETCDF4") as ds:
        ds.setncattr("time_step", int(cfg.time_res))
        ds.createDimension("channel", len(channels))
        ds.createDimension("latitude", height)
        ds.createDimension("longitude", width)

        channel_var = ds.createVariable("channel", str, ("channel",))
        channel_var[:] = np.asarray(channels, dtype=object)

        latitudes = np.linspace(90.0, -90.0, height, dtype=np.float32)
        longitudes = np.linspace(0.0, 360.0, width, endpoint=False, dtype=np.float32)
        ds.createVariable("latitude", "f4", ("latitude",))[:] = latitudes
        ds.createVariable("longitude", "f4", ("longitude",))[:] = longitudes

        fields = ds.createVariable(
            "fields",
            "f4",
            ("channel", "latitude", "longitude"),
            zlib=True,
            complevel=1,
            shuffle=True,
            fill_value=np.float32(0.0),
            chunksizes=_chunk_size(height, width),
        )
        fields.setncattr("time_step", int(cfg.time_res))


def _safe_remove(path):
    if os.path.lexists(path):
        os.remove(path)


def _safe_symlink(src, dst):
    if os.path.lexists(dst):
        return
    rel_src = os.path.relpath(src, os.path.dirname(dst))
    os.symlink(rel_src, dst)


def _write_stats(cfg):
    stats_dir = _resolve_path(CURRENT_DIR, cfg.stats_dir)
    os.makedirs(stats_dir, exist_ok=True)

    means = np.zeros((1, len(cfg.channels), 1, 1), dtype=np.float32)
    stds = np.ones((1, len(cfg.channels), 1, 1), dtype=np.float32)

    np.save(os.path.join(stats_dir, "global_means.npy"), means)
    np.save(os.path.join(stats_dir, "global_stds.npy"), stds)
    print(f"✅ Wrote zero/one stats to {stats_dir}")


def _write_metadata(cfg, years):
    data_dir = _resolve_path(CURRENT_DIR, cfg.data_dir)
    metadata = {
        "years": [str(year) for year in years],
        "variables": list(cfg.channels),
        "total_files": int(cfg.num_steps_per_year),
    }
    metadata_path = os.path.join(data_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Wrote metadata to {metadata_path}")


def generate_debug_sample(cfg):
    dataset_dir = _resolve_path(CURRENT_DIR, cfg.data_dir)
    data_root = os.path.join(dataset_dir, "data")
    os.makedirs(data_root, exist_ok=True)

    years = _collect_years(cfg)
    if not years:
        raise ValueError("No years found in config. Please set train_years/val_years/test_years.")

    real_year = years[0]
    real_dir = os.path.join(data_root, str(real_year))
    os.makedirs(real_dir, exist_ok=True)
    real_timestamp = _timestamp_strings(real_year, int(cfg.num_steps_per_year), int(cfg.time_res))[0]
    real_path = os.path.join(real_dir, f"{real_timestamp}.nc")
    _safe_remove(real_path)
    _create_sparse_zero_template(real_path, cfg)
    print(f"✅ Created sparse zero template: {real_path}")

    for year in years:
        year_dir = os.path.join(data_root, str(year))
        os.makedirs(year_dir, exist_ok=True)
        for timestamp in _timestamp_strings(year, int(cfg.num_steps_per_year), int(cfg.time_res)):
            dst = os.path.join(year_dir, f"{timestamp}.nc")
            if year == real_year and timestamp == real_timestamp:
                continue
            _safe_remove(dst)
            _safe_symlink(real_path, dst)

    print(f"✅ Created per-timestamp soft links for years: {years}")
    _write_metadata(cfg, years)
    _write_stats(cfg)


if __name__ == "__main__":
    cfg = YParams(os.path.join(CURRENT_DIR, "conf", "config.yaml"), "datapipe")
    generate_debug_sample(cfg.dataset)
