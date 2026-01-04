import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
import h5py
from tqdm import tqdm
import os
import json
import numpy as np
import h5py
import onnxruntime as ort
from onescience.utils.YParams import YParams



def data_prepare(date, channels, datapath):
    print('preparing data... ', end=' ')
    with open(f'{datapath}/metadata.json', "r") as f:
        metadata = json.load(f)

    variables = metadata['variables']
    channel_indices = [variables.index(v) for v in channels]
    with h5py.File(f'{datapath}/data/{date[:4]}/{date}.h5', "r") as f:
        data = f["fields"][:]
        data = data[channel_indices]
    print('done...')
    return data


def plot(date, label, pth_pred, onnx_pred, var, filename):
    # Plot for each A ID
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    plt.rcParams["font.size"] = 16

    # Row 1: True
    global_xtick_labels = ['180°W', '90°W', '0°', '90°E', '180°E']
    global_ytick_labels = ['90°S', '45°S', '0°', '45°N', '90°N']
    xticks = np.linspace(0, label.shape[-1] - 1, 5)
    yticks = np.linspace(0, label.shape[-2] - 1, 5)

    im0 = axes[0, 0].imshow(label, cmap='viridis')
    axes[0, 0].set_title(f'{date[:4]}-{date[4:6]}-{date[6:8]} {date[8:]} h')
    axes[0, 0].set_xlabel('Longitude')
    axes[0, 0].set_ylabel('Latitude')
    axes[0, 0].set_xticks(xticks)
    axes[0, 0].set_xticklabels(global_xtick_labels)
    axes[0, 0].set_yticks(yticks)
    axes[0, 0].set_yticklabels(global_ytick_labels)
    cbar0 = plt.colorbar(im0, ax=axes[0, 0], orientation='horizontal')

    # Row 2: Pred
    im1 = axes[0, 1].imshow(onnx_pred, cmap='viridis')
    axes[0, 1].set_title(f'Offical Prediction')
    axes[0, 1].set_xlabel('Longitude')
    axes[0, 1].set_ylabel('Latitude')
    axes[0, 1].set_xticks(xticks)
    axes[0, 1].set_xticklabels(global_xtick_labels)
    axes[0, 1].set_yticks(yticks)
    axes[0, 1].set_yticklabels(global_ytick_labels)
    cbar1 = plt.colorbar(im1, ax=axes[0, 1], orientation='horizontal')

    # Row 3: Diff
    im2 = axes[0, 2].imshow(label - onnx_pred, cmap='RdBu_r')
    rmse = np.sqrt(np.mean((label - onnx_pred) ** 2))
    axes[0, 2].set_title(f'Diff Distribution (rmse:{rmse: .2f})')
    axes[0, 2].set_xlabel('Longitude')
    axes[0, 2].set_ylabel('Latitude')
    axes[0, 2].set_xticks(xticks)
    axes[0, 2].set_xticklabels(global_xtick_labels)
    axes[0, 2].set_yticks(yticks)
    axes[0, 2].set_yticklabels(global_ytick_labels)
    cbar2 = plt.colorbar(im2, ax=axes[0, 2], orientation='horizontal')

    im3 = axes[1, 0].imshow(label, cmap='viridis')
    axes[1, 0].set_title(f'{date[:4]}-{date[4:6]}-{date[6:8]} {date[8:]} h')
    axes[1, 0].set_xlabel('Longitude')
    axes[1, 0].set_ylabel('Latitude')
    axes[1, 0].set_xticks(xticks)
    axes[1, 0].set_xticklabels(global_xtick_labels)
    axes[1, 0].set_yticks(yticks)
    axes[1, 0].set_yticklabels(global_ytick_labels)
    cbar0 = plt.colorbar(im3, ax=axes[1, 0], orientation='horizontal')

    # Row 2: Pred
    im4 = axes[1, 1].imshow(pth_pred, cmap='viridis')
    axes[1, 1].set_title(f'OneScience Prediction')
    axes[1, 1].set_xlabel('Longitude')
    axes[1, 1].set_ylabel('Latitude')
    axes[1, 1].set_xticks(xticks)
    axes[1, 1].set_xticklabels(global_xtick_labels)
    axes[1, 1].set_yticks(yticks)
    axes[1, 1].set_yticklabels(global_ytick_labels)
    cbar1 = plt.colorbar(im4, ax=axes[1, 1], orientation='horizontal')

    # Row 3: Diff
    im5 = axes[1, 2].imshow(label - pth_pred, cmap='RdBu_r')
    rmse = np.sqrt(np.mean((label - pth_pred) ** 2))
    axes[1, 2].set_title(f'Diff Distribution (rmse:{rmse: .2f})')
    axes[1, 2].set_xlabel('Longitude')
    axes[1, 2].set_ylabel('Latitude')
    axes[1, 2].set_xticks(xticks)
    axes[1, 2].set_xticklabels(global_xtick_labels)
    axes[1, 2].set_yticks(yticks)
    axes[1, 2].set_yticklabels(global_ytick_labels)
    cbar2 = plt.colorbar(im5, ax=axes[1, 2], orientation='horizontal')

    axes[0, 1].annotate(var, xy=(0.5, 1.5), xycoords='axes fraction', fontsize=20, ha='center')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


if __name__ == "__main__":
    config_file_path = "../conf/config.yaml"
    cfg = YParams(config_file_path, "model")
    ## DataLoader init
    cfg_data = YParams(config_file_path, "datapipe")
    channels = cfg_data.dataset.channels
    datapath = cfg_data.dataset.data_dir
    # notice that the date must 6 hours later to infer.py
    date = "2020010518"
    truth_data = data_prepare(date, channels, datapath)
    pth_pred = np.load(f'../result/output/{date}.npy')[0]
    onnx_pred = np.load(f'./output/onnx_output.npy')
    
    var = '2m_temperature' # 2m_temperature  geopotential_500  temperature_500
    var_index = cfg_data.dataset.channels.index(var)
    plot(date,
         truth_data[var_index], 
         pth_pred[var_index], 
         onnx_pred[var_index], 
         var, 
         f'{date}_{var}_compare.png')
    print(f'plot {var} compare result...')
    