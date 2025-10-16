import matplotlib.pyplot as plt
import os
import sys
import torch
import json
import numpy as np
import glob
import h5py
import time
from onescience.utils.fcn.YParams import YParams
from onescience.datapipes.climate import ERA5HDF5Datapipe


def get_error(cfg, mode, label_path, pred_path):
    year_splits = get_year_splits()
    test_year = year_splits['test']
    print('\n\n')
    print('-' * 40)
    print(f'testset total: {len(test_year)} years: test_year')
    print(f'now process {mode} result')
    print('-' * 40, '\n\n')
    stds = np.load(f'{cfg.stats_dir}/global_stds.npy')[0]
    means = np.load(f'{cfg.stats_dir}/global_means.npy')[0]
    
    total_rmse = 0
    channel_rmse = np.zeros(len(cfg.channels))
    for i in range(len(test_year)):
        year = test_year[i]
        print(f'process {pred_path}/{year}')

        label_data_list = sorted(glob.glob(os.path.join(label_path, 'data', str(year), "*.h5")))
        pred_data_list = sorted(glob.glob(os.path.join(pred_path, 'data', str(year), "*.h5")))
        print(f'year {year} has {len(label_data_list)} samples.')
        label_data_list = label_data_list[-len(pred_data_list):]
        for j in range(len(label_data_list)):
            with h5py.File(label_data_list[j], "r") as f:
                label = np.array(f["fields"][:])  # [N, H, W]
            with h5py.File(pred_data_list[j], "r") as f:
                pred = np.array(f["fields"][:])  # [N, H, W]
            label = label * stds + means
            pred = pred * stds + means
            # Compute RMSE per channel and total
            for k in range(len(cfg.channels)):
                rmse_i = np.sqrt(np.mean((label[k] - pred[k]) ** 2))
                channel_rmse[k] += rmse_i
                total_rmse += rmse_i
    channel_rmse /= (len(test_year) * len(label_data_list))
    for c in range(len(cfg.channels)):
        print(f"Var {cfg.channels[c]} RMSE: {channel_rmse[c]: .4f}, mean {means[i, 0, 0]: .4f}, stds {stds[i, 0, 0]: .4f}")
    
    year = test_year[0]
    label_data_list = sorted(glob.glob(os.path.join(label_path, 'data', str(year), "*.h5")))
    pred_data_list = sorted(glob.glob(os.path.join(pred_path, 'data', str(year), "*.h5")))
    # Generate random indices
    np.random.seed(42)
    sample_index = np.random.choice(len(label_data_list), 3, replace=False)
    channel_index = np.random.choice(len(cfg.channels), 3, replace=False)
    print(f"sample_index: {sample_index}")
    print(f"channel_index: {channel_index}")
    # Plot for each A ID
    for si in sample_index:
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        for i, ci in enumerate(channel_index):
            with h5py.File(label_data_list[si], "r") as f:
                label = np.array(f["fields"][ci])  # [N, H, W]
            with h5py.File(pred_data_list[si], "r") as f:
                pred = np.array(f["fields"][ci])  # [N, H, W]
            # Row 1: True
            global_xtick_labels = ['180°W', '90°W', '0°', '90°E', '180°E']
            global_ytick_labels = ['90°S', '45°S', '0°', '45°N', '90°N']
            xticks = np.linspace(0, label.shape[-1] - 1, 5)
            yticks = np.linspace(0, label.shape[-2] - 1, 5)

            im0 = axes[i, 0].imshow(label, cmap='viridis')
            axes[i, 0].set_title(f'Truth')
            axes[i, 0].set_xlabel('Longitude')
            axes[i, 0].set_ylabel('Latitude')
            axes[i, 0].set_xticks(xticks)
            axes[i, 0].set_xticklabels(global_xtick_labels)
            axes[i, 0].set_yticks(yticks)
            axes[i, 0].set_yticklabels(global_ytick_labels)
            cbar0 = plt.colorbar(im0, ax=axes[i, 0], orientation='horizontal')

            # Row 2: Pred
            im1 = axes[i, 1].imshow(pred, cmap='viridis')
            axes[i, 1].set_title(f'Pred')
            axes[i, 1].set_xlabel('Longitude')
            axes[i, 1].set_ylabel('Latitude')
            axes[i, 1].set_xticks(xticks)
            axes[i, 1].set_xticklabels(global_xtick_labels)
            axes[i, 1].set_yticks(yticks)
            axes[i, 1].set_yticklabels(global_ytick_labels)
            cbar1 = plt.colorbar(im1, ax=axes[i, 1], orientation='horizontal')

            # Row 3: Diff
            im2 = axes[i, 2].imshow(label - pred, cmap='RdBu_r')
            rmse = np.sqrt(np.mean((label - pred) ** 2))
            axes[i, 2].set_title(f'diff, RMSE={rmse: .2f}')
            axes[i, 2].set_xlabel('Longitude')
            axes[i, 2].set_ylabel('Latitude')
            axes[i, 2].set_xticks(xticks)
            axes[i, 2].set_xticklabels(global_xtick_labels)
            axes[i, 2].set_yticks(yticks)
            axes[i, 2].set_yticklabels(global_ytick_labels)
            cbar2 = plt.colorbar(im2, ax=axes[i, 2], orientation='horizontal')

            # Add row labels for each variable

            axes[i, 1].annotate(cfg.channels[i], xy=(0.5, 1.2), xycoords='axes fraction', fontsize=14, ha='center')
        os.makedirs('result/', exist_ok=True)
        plt.suptitle(f'sample {si} - True / Pred / Diff')
        plt.tight_layout()
        plt.savefig(f'result/{mode}_sample_{si}', dpi=300)


def get_year_splits():
    meta_path = os.path.join(cfg.data_dir, 'metadata.json')
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    years = list(map(int, metadata["years"]))
    y = sorted(years)
    if cfg.train_ratio + cfg.val_ratio + cfg.test_ratio == 1:
        n_train = int(len(y) * cfg.train_ratio)
        n_val = int(len(y) * cfg.val_ratio)
        year_splits = {
            "train": y[:n_train],
            "val": y[n_train:n_train + n_val],
            "test": y[n_train + n_val:]
        }
    elif cfg.train_ratio + cfg.val_ratio + cfg.test_ratio == len(y):
        n_train =  cfg.train_ratio
        n_val = cfg.val_ratio
        year_splits = {
            "train": y[:n_train],
            "val": y[n_train:n_train + n_val],
            "test": y[n_train + n_val:]
        }
    else:
        print('\n\n')
        print('-' * 30)
        print('Train/Val/Test settings must use ratio or digital numbers')
        print('If using ratio, please ensure the sum of all ratios equal to 1')
        print(f'If using digital number, please ensure the sum of number equal to total years {len(y)}')
        print(f'❌❌ Now settings are {cfg.train_ratio}-{cfg.val_ratio}-{cfg.test_ratio}, please check.')
        print('-' * 30)
        print('\n\n')
        exit()
        
    return year_splits


if __name__ == "__main__":
    current_path = os.getcwd()
    sys.path.append(current_path)
    config_file_path = os.path.join(current_path, "conf/config.yaml")
    cfg = YParams(config_file_path, "model")

    label_path = cfg.data_dir
    pred_path = './data/medium'

    get_error(cfg, 'short', label_path, pred_path)
    