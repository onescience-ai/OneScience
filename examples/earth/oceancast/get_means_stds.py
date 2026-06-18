import glob
from pathlib import Path

import h5py
import numpy as np

from onescience.utils.YParams import YParams


ROOT_DIR = Path(__file__).resolve().parent
CFG_DATA = YParams(str(ROOT_DIR / "conf" / "config.yaml"), "datapipe")
DATASET = CFG_DATA.dataset
OUTPUT_DIR = ROOT_DIR / DATASET.stats_dir
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def base_data_type(data_type):
    return data_type[:-9] if data_type.endswith("_Forecast") else data_type


DATA_TYPES = sorted(
    {
        *[
            base_data_type(item)
            for item in DATASET.input_types
            if not base_data_type(item).endswith("Sin") and not base_data_type(item).endswith("Cos")
        ],
        *[base_data_type(item) for item in DATASET.output_types],
    }
)


def data_dir_for(data_type):
    if data_type.startswith("Wind"):
        return ROOT_DIR / DATASET.wind_data_dir / data_type
    return ROOT_DIR / DATASET.ocean_data_dir / data_type


def crop_latitude_band(data):
    if data.shape[-2] == 170:
        return data[:, :-10]
    if data.shape[-2] == 180:
        return data[:, 10:-10]
    return data


def process_data():
    for data_type in DATA_TYPES:
        print(f"Processing type: {data_type}")
        all_data = []
        files = sorted(glob.glob(str(data_dir_for(data_type) / "*.h5")))
        for file_path in files:
            with h5py.File(file_path, "r") as h5_file:
                dataset = h5_file[list(h5_file.keys())[0]][:]
            dataset = crop_latitude_band(dataset)
            all_data.append(dataset[~np.isnan(dataset)])

        all_data = np.concatenate(all_data)
        global_mean = np.nanmean(all_data)
        global_std = np.nanstd(all_data)

        mean_matrix = np.full(DATASET.img_size, global_mean, dtype=np.float32)
        std_matrix = np.full(DATASET.img_size, global_std, dtype=np.float32)
        np.save(OUTPUT_DIR / f"{data_type}_means.npy", mean_matrix)
        np.save(OUTPUT_DIR / f"{data_type}_stds.npy", std_matrix)
        print(f"{data_type}: mean={global_mean:.6f}, std={global_std:.6f}")


if __name__ == "__main__":
    process_data()
