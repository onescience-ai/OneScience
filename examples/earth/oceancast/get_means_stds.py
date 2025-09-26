import glob
import os
import sys

import h5py
import numpy as np

from onescience.utils.fcn.YParams import YParams

current_path = os.getcwd()
sys.path.append(current_path)
config_file_path = os.path.join(
    current_path, "conf/oceancast.yaml")
params = YParams(config_file_path, "afno_backbone")
# If forecast wave parameters, use following types
types = ["Wave_Height", "Wave_Direction",
         "Wave_Period", "Wind_Strength"]

# If forecast sea surface parameters, use following types
# types = ['Ocean_SSH', 'Ocean_SSS', 'Ocean_SST', 'Wind_Strength']

output_dir = "means_stds"
os.makedirs(output_dir, exist_ok=True)
data_dir = params.ocean_data_path


def calculate_global_mean_std(data_sum, count, data_sum_sq):
    global_mean = data_sum / count
    global_std = np.sqrt(
        data_sum_sq / count - global_mean**2)
    return global_mean, global_std


def process_data():
    for data_type in types:
        print(f"Processing type: {data_type}")
        all_data = []
        files = glob.glob(os.path.join(
            data_dir, data_type, "*.h5"))
        for file_path in files:
            with h5py.File(file_path, "r") as f:
                dataset_name = list(f.keys())[0]
                data = f[dataset_name][:]
                if data.shape[-2] == 170:
                    data = data[:, :-10]
                if data.shape[-2] == 180:
                    data = data[:, 10:-10]
                all_data.append(data[~np.isnan(data)])

        all_data = np.concatenate(all_data)
        global_mean = np.nanmean(all_data)
        global_std = np.nanstd(all_data)
        print(data_type, global_mean, global_std)
        mean_matrix = np.full((160, 360), global_mean)
        std_matrix = np.full((160, 360), global_std)
        np.save(os.path.join(
            output_dir, f"{data_type}_means.npy"), mean_matrix)
        np.save(os.path.join(
            output_dir, f"{data_type}_stds.npy"), std_matrix)
        print(
            f"Saved {data_type} mean and std to {output_dir}")


if __name__ == "__main__":
    process_data()
