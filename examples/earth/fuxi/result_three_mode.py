import matplotlib.pyplot as plt
import os
import sys
import torch
import numpy as np
import glob
import h5py
import time
from onescience.utils.fcn.YParams import YParams


def get_error(mode, pred_path, label):
    stds = np.load(f'{cfg.stats_dir}/global_stds.npy')[0]
    means = np.load(f'{cfg.stats_dir}/global_means.npy')[0]
    pred_list = []

    for k in range(len(pred_path)):
        st = time.perf_counter()
        print(f'process {pred_path[k]}')

        with h5py.File(pred_path[k], 'r') as f:
            pred = np.array(f["fields"][:])
        pred_list.append(pred)
        print(f'read {pred_path[k]}, shape: {pred.shape}, cost {time.perf_counter() - st: .2f}s')

    # 合并所有预测和标签
    pred = np.stack(pred_list, axis=0)
    print(pred.shape, label.shape)
    

    assert pred.shape[0] == label.shape[0], f"The files are not identical in pred and truth, please check. Details: pred-{pred}; truth-{truth}"
    pred = pred.reshape(-1, pred.shape[-3], pred.shape[-2], pred.shape[-1])
    label = label.reshape(-1, label.shape[-3], label.shape[-2], label.shape[-1])
    label = label[-pred.shape[0]:]
    print(f"✅ Pred files processed.  Total {len(label_path)} files. Final shapes: pred={pred.shape}, label={label.shape}")
    label = label * stds + means
    pred = pred * stds + means
    print('Anti-normalization and get original data...')
    # Compute RMSE per channel and total
    total_rmse = 0
    num_channels = label.shape[1]
    for i in range(num_channels):
        rmse_i = np.sqrt(np.mean((label[:, i] - pred[:, i]) ** 2))
        print(f"Channel {cfg.channels[i]} RMSE: {rmse_i: .4f}, mean {means[i, 0, 0]: .4f}, stds {stds[i, 0, 0]: .4f}")
        total_rmse += rmse_i

    mean_rmse = total_rmse / num_channels
    print(f"Mean RMSE over {num_channels} channels: {mean_rmse: .4f}")

    # Generate random indices
    np.random.seed(42)
    sample_index = np.random.choice(label.shape[0], 3, replace=False)
    channel_index = np.random.choice(label.shape[1], 3, replace=False)
    print(f"sample_index: {sample_index}")
    print(f"channel_index: {channel_index}")

    # Plot for each A ID
    for si in sample_index:
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))

        for i, ci in enumerate(channel_index):
            # Row 1: True
            global_xtick_labels = ['180°W', '90°W', '0°', '90°E', '180°E']
            global_ytick_labels = ['90°S', '45°S', '0°', '45°N', '90°N']
            xticks = np.linspace(0, label[si, ci].shape[-1] - 1, 5)
            yticks = np.linspace(0, label[si, ci].shape[-2] - 1, 5)

            im0 = axes[i, 0].imshow(label[si, ci], cmap='viridis')
            axes[i, 0].set_title(f'Truth')
            axes[i, 0].set_xlabel('Longitude')
            axes[i, 0].set_ylabel('Latitude')
            axes[i, 0].set_xticks(xticks)
            axes[i, 0].set_xticklabels(global_xtick_labels)
            axes[i, 0].set_yticks(yticks)
            axes[i, 0].set_yticklabels(global_ytick_labels)
            cbar0 = plt.colorbar(im0, ax=axes[i, 0], orientation='horizontal')

            # Row 2: Pred
            im1 = axes[i, 1].imshow(pred[si, ci], cmap='viridis')
            axes[i, 1].set_title(f'Pred')
            axes[i, 1].set_xlabel('Longitude')
            axes[i, 1].set_ylabel('Latitude')
            axes[i, 1].set_xticks(xticks)
            axes[i, 1].set_xticklabels(global_xtick_labels)
            axes[i, 1].set_yticks(yticks)
            axes[i, 1].set_yticklabels(global_ytick_labels)
            cbar1 = plt.colorbar(im1, ax=axes[i, 1], orientation='horizontal')

            # Row 3: Diff
            im2 = axes[i, 2].imshow(label[si, ci] - pred[si, ci], cmap='RdBu_r')
            rmse = np.sqrt(np.mean((label[si, ci] - pred[si, ci]) ** 2))
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

        plt.suptitle(f'sample {si} - True / Pred / Diff')
        plt.tight_layout()
        plt.savefig(f'result/{mode}_sample_{si}', dpi=300)

if __name__ == "__main__":
    current_path = os.getcwd()
    sys.path.append(current_path)
    config_file_path = os.path.join(current_path, "conf/config.yaml")
    cfg = YParams(config_file_path, "fuxi")

    label_list = []
    label_path = glob.glob(f"{cfg.test_data_dir}/*.h5")
    for k in range(len(label_path)):
        print(f'process {label_path[k]}')
        st = time.perf_counter()
        with h5py.File(label_path[k], 'r') as f:
            label = np.array(f["fields"][:])
        label_list.append(label)
        print(f'read {label_path[k]}, shape: {label.shape}, cost {time.perf_counter() - st: .2f}s')

    # 合并所有预测和标签
    label_all = np.stack(label_list, axis=0)

    print(f"Ground-truth files processed. Total {len(label_path)} files. shapes: label={label_all.shape}")

    short_pred_files = sorted(glob.glob(f"{cfg.test_data_dir}/medium/*.h5"))
    medium_pred_files = sorted(glob.glob(f"{cfg.test_data_dir}/long/*.h5"))
    long_pred_files = sorted(glob.glob(f"{cfg.test_data_dir}/final/*.h5"))


    get_error('short', short_pred_files, label_all)
    get_error('medium', medium_pred_files, label_all)
    get_error('long', long_pred_files, label_all)