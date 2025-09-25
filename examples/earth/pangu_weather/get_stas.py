import glob
import os
import sys

import h5py
import numpy as np

from onescience.utils.fcn.YParams import YParams

current_path = os.getcwd()
sys.path.append(current_path)
config_file_path = os.path.join(current_path, "conf/config.yaml")
cfg = YParams(config_file_path, "pangu")

timelabel = glob.glob(f"{cfg.train_data_dir}/*.h5")
global_means = np.zeros((1, len(cfg.channels), 1, 1))
global_stds = np.zeros((1, len(cfg.channels), 1, 1))
print("total ", len(cfg.channels), " channels")
for tl in timelabel:
    print(f"process: {tl}")
    with h5py.File(tl, "r") as f:
        global_means += np.mean(f["fields"], keepdims=True, axis=(0, 2, 3))
        global_stds += np.var(f["fields"], keepdims=True, axis=(0, 2, 3))

global_means = global_means / len(timelabel)
global_stds = np.sqrt(global_stds / len(timelabel))
np.save(f"{cfg.stats_dir}/global_means.npy", global_means)
np.save(f"{cfg.stats_dir}/global_stds.npy", global_stds)

print(global_means.shape, global_stds.shape)
for i in range(len(cfg.channels)):
    print(f"channel {cfg.channels[i]} means: {global_means[0, i, 0, 0]: .4f}")
for i in range(len(cfg.channels)):
    print(f"channel {cfg.channels[i]} stds:  {global_stds[0, i, 0, 0]: .4f}")
