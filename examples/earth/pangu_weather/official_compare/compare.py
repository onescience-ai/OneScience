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
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # 坐标轴标签
    xtick_labels = ['180°W', '90°W', '0°', '90°E', '180°E']
    ytick_labels = ['90°S', '45°S', '0°', '45°N', '90°N']
    xticks = np.linspace(0, label.shape[-1] - 1, 5)
    yticks = np.linspace(0, label.shape[-2] - 1, 5)
    
    # 计算统一色条范围
    vmin = min(label.min(), pth_pred.min(), onnx_pred.min())
    vmax = max(label.max(), pth_pred.max(), onnx_pred.max())
    
    # 计算差异
    onnx_diff = label - onnx_pred
    pth_diff = label - pth_pred
    onnx_rmse = np.sqrt(np.mean(onnx_diff ** 2))
    pth_rmse = np.sqrt(np.mean(pth_diff ** 2))
    diff_abs_max = max(np.abs(onnx_diff).max(), np.abs(pth_diff).max())
    
    # 格式化日期
    date_str = f'{date[:4]}-{date[4:6]}-{date[6:8]} {date[8:]}h'
    
    # 绑图配置：两行分别对应 Official 和 OneScience
    row_configs = [
        {'pred': onnx_pred, 'diff': onnx_diff, 'rmse': onnx_rmse, 'name': 'Official'},
        {'pred': pth_pred,  'diff': pth_diff,  'rmse': pth_rmse,  'name': 'OneScience'},
    ]
    
    for row, cfg in enumerate(row_configs):
        # 每行三列的配置
        col_configs = [
            {'data': label,      'title': f'Truth ({date_str})', 'cmap': 'viridis', 'vmin': vmin, 'vmax': vmax},
            {'data': cfg['pred'], 'title': f'{cfg["name"]} Prediction', 'cmap': 'viridis', 'vmin': vmin, 'vmax': vmax},
            {'data': cfg['diff'], 'title': f'Difference (RMSE={cfg["rmse"]:.2f})', 'cmap': 'RdBu_r', 'vmin': -diff_abs_max, 'vmax': diff_abs_max},
        ]
        
        for col, ccfg in enumerate(col_configs):
            ax = axes[row, col]
            im = ax.imshow(ccfg['data'], cmap=ccfg['cmap'], vmin=ccfg['vmin'], vmax=ccfg['vmax'])
            ax.set_title(ccfg['title'], fontsize=12, pad=4)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_xticks(xticks)
            ax.set_xticklabels(xtick_labels)
            ax.set_yticks(yticks)
            ax.set_yticklabels(ytick_labels)
            plt.colorbar(im, ax=ax, orientation='horizontal')
    
    # 总标题
    fig.suptitle(var, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def get_rmse(channel_indices):
    channel_rmse = np.zeros(len(channel_indices))
    pth_rmse =  np.load('../result/rmse.npy')
    onnx_rmse = np.load('./result/rmse.npy')
    
    channels = [cfg_data.dataset.channels[i] for i in range(len(channel_indices))]
    w = 24  # 最长 channel 名宽度
    
    # 表头
    print(f"┌{'─' * (w + 2)}┬{'─' * 14}┬{'─' * 14}┐")
    print(f"│ {'Channel':<{w}} │ {'OneScience':>12} │ {'Official':>12} │")
    print(f"├{'─' * (w + 2)}┼{'─' * 14}┼{'─' * 14}┤")
    
    # 数据行
    for i, ch in enumerate(channels):
        print(f"│ {ch:<{w}} │ {pth_rmse[i]:>12.4f} │ {onnx_rmse[i]:>12.4f} │")


if __name__ == "__main__":
    config_file_path = "../conf/config.yaml"
    cfg = YParams(config_file_path, "model")
    ## DataLoader init
    cfg_data = YParams(config_file_path, "datapipe")
    channels = cfg_data.dataset.channels
    datapath = cfg_data.dataset.data_dir

    meta_path = os.path.join(cfg.data_dir, 'metadata.json')
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    variables = metadata['variables']
    channel_indices = [variables.index(v) for v in cfg.channels]
    get_rmse(channel_indices)

    date = "2020010212"
    truth_data = data_prepare(date, channels, datapath)
    pth_pred = np.load(f'../result/output/{date}.npy')[0]
    onnx_pred = np.load(f'./result/output/{date}.npy')
    
    var = '2m_temperature' # 2m_temperature  geopotential_500  temperature_500
    var_index = cfg_data.dataset.channels.index(var)
    plot(date,
         truth_data[var_index], 
         pth_pred[var_index], 
         onnx_pred[var_index], 
         var, 
         f'{date}_{var}_compare1.png')
    print(f'plot {var} compare result...')
    