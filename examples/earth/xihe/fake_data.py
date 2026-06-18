import json
import os

import h5py
import numpy as np

from onescience.utils.YParams import YParams


DATASET_DIMS = {"T": 10, "H": 2041, "W": 4320, "time_step": 24}


def generate_fake_h5(data_dir, var_names, years, dims):
    os.makedirs(os.path.join(data_dir, "data"), exist_ok=True)
    t_dim, c_dim = dims["T"], len(var_names)
    h_dim, w_dim = dims["H"], dims["W"]

    for year in years:
        path = os.path.join(data_dir, "data", f"{year}.h5")
        with h5py.File(path, "w") as f:
            ds = f.create_dataset(
                "fields",
                shape=(t_dim, c_dim, h_dim, w_dim),
                dtype="float32",
                chunks=(1, c_dim, h_dim, w_dim),
                fillvalue=0.0,
            )
            ds.attrs["variables"] = var_names
            ds.attrs["time_step"] = dims["time_step"]

        size_kb = os.path.getsize(path) / 1024
        print(
            f"  {year}.h5  shape=({t_dim},{c_dim},{h_dim},{w_dim})  "
            f"logical={t_dim * c_dim * h_dim * w_dim * 4 / 1024**3:.1f}GB  actual={size_kb:.1f}KB"
        )


def generate_metadata(data_dir, var_names, years):
    metadata = {
        "years": [str(year) for year in years],
        "variables": var_names,
    }
    with open(os.path.join(data_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  metadata saved -> {os.path.join(data_dir, 'metadata.json')}")


def generate_stats(stats_dir, n_vars):
    os.makedirs(stats_dir, exist_ok=True)
    shape = (1, n_vars, 1, 1)
    np.save(os.path.join(stats_dir, "global_means.npy"), np.zeros(shape, dtype=np.float32))
    np.save(os.path.join(stats_dir, "global_stds.npy"), np.ones(shape, dtype=np.float32))
    print(f"  stats saved -> {stats_dir}")


def generate_mask(save_path, shape, one_ratio=0.7, seed=42):
    np.random.seed(seed)
    mask = (np.random.rand(*shape) < one_ratio).astype(np.float32)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, mask)
    print(f"  mask saved -> {save_path}")


def main():
    cfg_model = YParams("conf/config.yaml", "model")
    cfg_datapipe = YParams("conf/config.yaml", "datapipe")

    if cfg_datapipe.dataset.data_dir.startswith("/public/onestore"):
        print("Please check config and ensure local test paths are used.")
        exit()

    years = (
        cfg_datapipe.dataset.train_time
        + cfg_datapipe.dataset.val_time
        + cfg_datapipe.dataset.test_time
    )
    channels = cfg_datapipe.dataset.channels

    generate_fake_h5(cfg_datapipe.dataset.data_dir, channels, years, DATASET_DIMS)
    generate_metadata(cfg_datapipe.dataset.data_dir, channels, years)
    generate_stats(cfg_datapipe.dataset.stats_dir, len(channels))
    generate_mask(cfg_model.mask, (DATASET_DIMS["H"], DATASET_DIMS["W"]))

    print("\n✅ Fake CMEMS datasets generated.")


if __name__ == "__main__":
    main()
