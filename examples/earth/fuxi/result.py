import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
import h5py
from tqdm import tqdm
from onescience.utils.fcn.YParams import YParams

def get_metadata(mode, cfg):
    meta_path = os.path.join(cfg.data_dir, 'metadata.json')
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    years = list(map(int, metadata["years"]))
    variables = metadata['variables']
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

    selected_years = year_splits['test']
    total_files = []
    for year in selected_years:
        if mode == 'base':
            path = os.path.join(cfg.data_dir, 'data', str(year))
        else:
            path = os.path.join(f'./result/{mode}', 'data', str(year)) 
        files = sorted(os.listdir(path))
        samples_per_year = len(files) - 1
        total_files.extend(files[-samples_per_year:])
    
    channel_indices = [variables.index(v) for v in cfg.channels]
    return total_files, channel_indices


def get_rmse(mode, total_files, channel_indices):
    total_rmse = 0
    channel_rmse = np.zeros(len(channel_indices))
    if not os.path.exists('result/rmse.npy'):
        for file in tqdm(total_files, unit="files"):
            if mode == 'base':
                h5file = f'{cfg_data.dataset.data_dir}/data/{file[:4]}/{file}'
            else:
                h5file = f'{cfg_data.dataset.data_dir}/data/{file[:4]}/{file[:-4]}.h5'
            with h5py.File(h5file, "r") as f:
                label = f["fields"][:]  # [N, H, W]
                label = label[channel_indices]
            pred = np.load(f'result/{mode}/data/{file[:4]}/{file}').squeeze()
            channel_rmse += np.sqrt(np.mean((label - pred) ** 2, axis=(1, 2)))
        channel_rmse /= len(total_files)
        np.save(f'result/{mode}_rmse.npy', channel_rmse)
    else:
        channel_rmse = np.load('result/rmse.npy')
    for i in range(len(channel_indices)):
        print(f"📂 Channel: {cfg_data.dataset.channels[i]} RMSE: {channel_rmse[i]: .4f},")
    print(f"✅ Avg RMSE is : {np.mean(channel_rmse): .4f}")


def plot(label, pred, var, filename):
    # Plot for each A ID
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # Row 1: True
    global_xtick_labels = ['180°W', '90°W', '0°', '90°E', '180°E']
    global_ytick_labels = ['90°S', '45°S', '0°', '45°N', '90°N']
    xticks = np.linspace(0, label.shape[-1] - 1, 5)
    yticks = np.linspace(0, label.shape[-2] - 1, 5)

    im0 = axes[0].imshow(label, cmap='viridis')
    axes[0].set_title(f'Truth')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    axes[0].set_xticks(xticks)
    axes[0].set_xticklabels(global_xtick_labels)
    axes[0].set_yticks(yticks)
    axes[0].set_yticklabels(global_ytick_labels)
    cbar0 = plt.colorbar(im0, ax=axes[0], orientation='horizontal')

    # Row 2: Pred
    im1 = axes[1].imshow(pred, cmap='viridis')
    axes[1].set_title(f'Pred')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    axes[1].set_xticks(xticks)
    axes[1].set_xticklabels(global_xtick_labels)
    axes[1].set_yticks(yticks)
    axes[1].set_yticklabels(global_ytick_labels)
    cbar1 = plt.colorbar(im1, ax=axes[1], orientation='horizontal')

    # Row 3: Diff
    im2 = axes[2].imshow(label - pred, cmap='RdBu_r')
    rmse = np.sqrt(np.mean((label - pred) ** 2))
    axes[2].set_title(f'diff, RMSE={rmse: .2f}')
    axes[2].set_xlabel('Longitude')
    axes[2].set_ylabel('Latitude')
    axes[2].set_xticks(xticks)
    axes[2].set_xticklabels(global_xtick_labels)
    axes[2].set_yticks(yticks)
    axes[2].set_yticklabels(global_ytick_labels)
    cbar2 = plt.colorbar(im2, ax=axes[2], orientation='horizontal')
    # Add row labels for each variable
    axes[1].annotate(var, xy=(0.5, 1.2), xycoords='axes fraction', fontsize=14, ha='center')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: input the mode: : base, short, medium, or long...")
        sys.exit(1)
    
    mode = sys.argv[1]
    if mode not in ['base', 'short', 'medium', 'long']:
        print(f'❌ ❌ Please input the mode: base, short, medium, or long...')
        exit()

    current_path = os.getcwd()
    sys.path.append(current_path)

    config_file_path = os.path.join(current_path, 'conf/config.yaml')
    cfg = YParams(config_file_path, 'model')
    cfg_data = YParams(config_file_path, "datapipe")
    total_files, channel_indices = get_metadata(mode, cfg_data.dataset)
    # Load data
    # Compute RMSE per channel and total
    get_rmse(mode, total_files, channel_indices)

    ##### You can choose the date to plot #####
    total_files = ['2019010106.h5', '2019012306.h5', '2020020806.h5']
    channel_index = [cfg_data.dataset.channels.index(v) for v in ['geopotential_500', 'temperature_500']]
    selected_files = total_files

    ##### Or use random index to plot #####
    # np.random.seed(42) # use a fix seed ensure to get same result
    # sample_index = np.random.choice(len(total_files), 3, replace=False)
    # channel_index = np.random.choice(len(cfg_data.dataset.channels), 3, replace=False)
    # selected_files = [total_files[int(i)] for i in sample_index]
    
    selected_var = [cfg_data.dataset.channels[int(i)] for i in channel_index]
    print(f"seleted date: {selected_files}")
    print(f"selected channels: {selected_var}")
    for file in selected_files:
        with h5py.File(f'{cfg_data.dataset.data_dir}/data/{file[:4]}/{file}', "r") as f:
            label = f["fields"][:]  # [N, H, W]
            label = label[channel_indices]
            # label = label[:, :-1, :]
            label = label[channel_index]
        pred = np.load(f'result/{mode}/data/{file[:4]}/{file[:-3]}.npy').squeeze()

        for i in range(len(selected_var)):
            filename = f'./result/{mode}_{file[:-3]}_{selected_var[i]}.png'
            print(f'✅plot {filename}')
            plot(label[i], pred[i], selected_var[i], filename)