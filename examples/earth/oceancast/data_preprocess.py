"""
This Code calculates the sin/cos of wave direction for a better representation.
Besides, this code also calculates the sin/cos/strength of wind for model to learn the effect of wind.

"""

import os
import sys

import h5py
import numpy as np

from onescience.utils.fcn.YParams import YParams

current_path = os.getcwd()
sys.path.append(current_path)
config_file_path = os.path.join(current_path, "conf/oceancast.yaml")
params = YParams(config_file_path, "afno_backbone")

# replace the following path of the data
wind_u_dir = f"{params.wind_uv_path}/Wind_U10"
wind_v_dir = f"{params.wind_uv_path}/Wind_V10"

# replace the following path of the data storage
Wind_Cos = f"{params.wind_data_path}/Wind_Cos"
Wind_Sin = f"{params.wind_data_path}/Wind_Sin"
Wind_Strength = f"{params.wind_data_path}/Wind_Strength"
os.makedirs(Wind_Cos, exist_ok=True)
os.makedirs(Wind_Sin, exist_ok=True)
os.makedirs(Wind_Strength, exist_ok=True)


def process_all_years():

    wind_u_files = sorted([f for f in os.listdir(f"{wind_u_dir}") if f.endswith(".h5")])
    for year_file in wind_u_files:
        wind_u_path = os.path.join(wind_u_dir, year_file)
        wind_v_path = os.path.join(wind_v_dir, year_file)

        if os.path.exists(f"{Wind_Strength}/{year_file}"):
            print(f"Skipping {Wind_Strength}/{year_file}, files already exist.")
            continue

        with h5py.File(wind_u_path, "r") as u_file, h5py.File(
            wind_v_path, "r"
        ) as v_file:
            print(f"process {year_file}")
            wind_u = u_file[list(u_file.keys())[0]][:]
            wind_v = v_file[list(v_file.keys())[0]][:]

            wind_strength = np.sqrt(wind_u**2 + wind_v**2)
            wind_direction = np.arctan2(wind_v, wind_u)
            wind_sin = np.sin(wind_direction)
            wind_cos = np.cos(wind_direction)

            wind_sin[np.isnan(wind_direction)] = 0
            wind_cos[np.isnan(wind_direction)] = 0

        with h5py.File(f"{Wind_Cos}/{year_file}", "w") as h5file:
            h5file.create_dataset("fields", data=wind_sin)
        with h5py.File(f"{Wind_Sin}/{year_file}", "w") as h5file:
            h5file.create_dataset("fields", data=wind_cos)
        with h5py.File(f"{Wind_Strength}/{year_file}", "w") as h5file:
            h5file.create_dataset("fields", data=wind_strength)

    print("all data process done...")


if __name__ == "__main__":
    process_all_years()
