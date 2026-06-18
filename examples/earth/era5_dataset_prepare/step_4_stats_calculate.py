import os

import h5py
import numpy as np


DATA_DIR = "./data"
OUTPUT_DIR = "./stats"
TIME_CHUNK_STEPS = 4
CHANNEL_CHUNK_SIZE = 8


def _decode_variables(values):
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values]


def calculate_stats():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    h5_files = sorted(
        os.path.join(DATA_DIR, name)
        for name in os.listdir(DATA_DIR)
        if name.endswith(".h5")
    )
    if not h5_files:
        raise FileNotFoundError(f"No yearly HDF5 files found under {DATA_DIR}")

    with h5py.File(h5_files[0], "r") as f:
        fields = f["fields"]
        variables = _decode_variables(fields.attrs["variables"])
        num_vars = fields.shape[1]

    sum_vals = np.zeros(num_vars, dtype=np.float64)
    sum_sq_vals = np.zeros(num_vars, dtype=np.float64)
    count = np.zeros(num_vars, dtype=np.int64)

    for file_path in h5_files:
        with h5py.File(file_path, "r") as f:
            fields = f["fields"]
            for start in range(0, fields.shape[0], TIME_CHUNK_STEPS):
                end = min(start + TIME_CHUNK_STEPS, fields.shape[0])
                for channel_start in range(0, num_vars, CHANNEL_CHUNK_SIZE):
                    channel_end = min(channel_start + CHANNEL_CHUNK_SIZE, num_vars)
                    data = np.asarray(
                        fields[start:end, channel_start:channel_end],
                        dtype=np.float64,
                    )
                    flat = data.transpose(1, 0, 2, 3).reshape(channel_end - channel_start, -1)
                    valid = ~np.isnan(flat)
                    values = np.where(valid, flat, 0.0)
                    sum_vals[channel_start:channel_end] += values.sum(axis=1)
                    sum_sq_vals[channel_start:channel_end] += (values * values).sum(axis=1)
                    count[channel_start:channel_end] += valid.sum(axis=1)
        print(f"Processed {file_path}")

    means = sum_vals / np.maximum(count, 1)
    stds = np.sqrt(np.maximum(sum_sq_vals / np.maximum(count, 1) - means ** 2, 0.0))

    np.save(
        os.path.join(OUTPUT_DIR, "global_means.npy"),
        means.reshape(1, num_vars, 1, 1).astype(np.float32),
    )
    np.save(
        os.path.join(OUTPUT_DIR, "global_stds.npy"),
        stds.reshape(1, num_vars, 1, 1).astype(np.float32),
    )

    for idx, var_name in enumerate(variables):
        print(f"{var_name}: mean={means[idx]:.6f}, std={stds[idx]:.6f}")
    print(f"Saved stats under {OUTPUT_DIR}")


if __name__ == "__main__":
    calculate_stats()
