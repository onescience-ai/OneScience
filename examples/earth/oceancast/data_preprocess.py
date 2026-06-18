import os
from pathlib import Path

import h5py
import numpy as np

from onescience.utils.YParams import YParams


ROOT_DIR = Path(__file__).resolve().parent
CFG_DATA = YParams(str(ROOT_DIR / "conf" / "config.yaml"), "datapipe")
DATASET = CFG_DATA.dataset

wind_u_dir = ROOT_DIR / DATASET.wind_uv_dir / "Wind_U10"
wind_v_dir = ROOT_DIR / DATASET.wind_uv_dir / "Wind_V10"
wind_cos_dir = ROOT_DIR / DATASET.wind_data_dir / "Wind_Cos"
wind_sin_dir = ROOT_DIR / DATASET.wind_data_dir / "Wind_Sin"
wind_strength_dir = ROOT_DIR / DATASET.wind_data_dir / "Wind_Strength"

os.makedirs(wind_cos_dir, exist_ok=True)
os.makedirs(wind_sin_dir, exist_ok=True)
os.makedirs(wind_strength_dir, exist_ok=True)


def process_all_years():
    wind_u_files = sorted([file_name for file_name in os.listdir(wind_u_dir) if file_name.endswith(".h5")])
    for year_file in wind_u_files:
        output_files = [wind_cos_dir / year_file, wind_sin_dir / year_file, wind_strength_dir / year_file]
        if all(file_path.exists() for file_path in output_files):
            print(f"Skipping {year_file}, files already exist.")
            continue

        with h5py.File(wind_u_dir / year_file, "r") as u_file, h5py.File(wind_v_dir / year_file, "r") as v_file:
            print(f"Processing {year_file}")
            wind_u = u_file[list(u_file.keys())[0]][:]
            wind_v = v_file[list(v_file.keys())[0]][:]

        wind_strength = np.sqrt(wind_u ** 2 + wind_v ** 2)
        wind_direction = np.arctan2(wind_v, wind_u)
        wind_sin = np.sin(wind_direction)
        wind_cos = np.cos(wind_direction)

        wind_sin[np.isnan(wind_direction)] = 0
        wind_cos[np.isnan(wind_direction)] = 0

        with h5py.File(wind_cos_dir / year_file, "w") as h5_file:
            h5_file.create_dataset("fields", data=wind_cos)
        with h5py.File(wind_sin_dir / year_file, "w") as h5_file:
            h5_file.create_dataset("fields", data=wind_sin)
        with h5py.File(wind_strength_dir / year_file, "w") as h5_file:
            h5_file.create_dataset("fields", data=wind_strength)

    print("All data processed.")


if __name__ == "__main__":
    process_all_years()
