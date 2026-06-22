import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import glob
import h5py
from tqdm import tqdm
import os
import numpy as np
import h5py
import onnxruntime as ort
from datetime import datetime
from onescience.utils.YParams import YParams



def data_prepare(date, channels, datapath):
    h5_files = sorted(glob.glob(os.path.join(datapath, "data", "*.h5")))
    with h5py.File(h5_files[0], "r") as f:
        ds = f["fields"]
        variables = [v.decode() if isinstance(v, bytes) else v for v in ds.attrs["variables"]]
        time_step = int(ds.attrs["time_step"])
    channel_indices = [variables.index(v) for v in channels]
    dt = datetime.strptime(date, "%Y%m%d%H")
    year_start = datetime(dt.year, 1, 1)
    step_idx = int(((dt - year_start).total_seconds() / 3600) / time_step)
    with h5py.File(os.path.join(datapath, "data", f"{date[:4]}.h5"), "r") as f:
        data = f["fields"][step_idx]
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


def show_compare(channel_indices):
    pth_rmse =  np.load('../result/rmse.npy')
    onnx_rmse = np.load('./result/rmse.npy')

    pth_acc = np.load('../result/acc.npy')
    onnx_acc = np.load('./result/acc.npy')  
    
    channels = [cfg_data.dataset.channels[i] for i in range(len(channel_indices))]
    w = 24  # 最长 channel 名宽度
    
    # 表头
    print(f"┌{'─' * (w + 2)}┬{'─' * 14}┬{'─' * 14}┬{'─' * 14}┬{'─' * 14}┐")
    print(f"│ {'Channel':<{w}} │ {'OneSci RMSE':>12} │ {'Official RMSE':>12} │ {'OneSci ACC':>12} │ {'Official ACC':>12} │")
    print(f"├{'─' * (w + 2)}┼{'─' * 14}┼{'─' * 14}┼{'─' * 14}┼{'─' * 14}┤")

    # 数据行
    for i, ch in enumerate(channels):
        print(f"│ {ch:<{w}} │ {pth_rmse[i]:>12.4f} │ {onnx_rmse[i]:>12.4f} │ {pth_acc[i]:>12.4f} │ {onnx_acc[i]:>12.4f} │")
    print(f"├{'─' * (w + 2)}┼{'─' * 14}┼{'─' * 14}┼{'─' * 14}┼{'─' * 14}┤")
    print(f"│ {'Average':<{w}} │ {np.mean(pth_rmse):>12.4f} │ {np.mean(onnx_rmse):>12.4f} │ {np.mean(pth_acc):>12.4f} │ {np.mean(onnx_acc):>12.4f} │")
    print(f"└{'─' * (w + 2)}┴{'─' * 14}┴{'─' * 14}┴{'─' * 14}┴{'─' * 14}┘")


def plot_rmse_comparison(channel_indices, filename='./result/rmse_comparison.png'):
    channels = [cfg_data.dataset.channels[i] for i in range(len(channel_indices))]
    pth_rmse =  np.load('../result/rmse.npy')
    onnx_rmse = np.load('./result/rmse.npy')
    fig, ax = plt.subplots(figsize=(15, 5))
    
    x = np.arange(len(channels))
    colors = {'pth': '#2563EB', 'onnx': '#EA580C'}

    # 绑定折线图
    ax.plot(x, pth_rmse, color=colors['pth'], linewidth=1.5, label=f'OneScience (avg rmse: {np.mean(pth_rmse):.2f})', marker='o', markersize=4)
    ax.plot(x, onnx_rmse, color=colors['onnx'], linewidth=1.5, label=f'Official (avg rmse: {np.mean(onnx_rmse):.2f})', marker='s', markersize=4)

    # 坐标轴设置
    ax.set_ylabel('RMSE (log scale)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(channels, rotation=45, ha='right', fontsize=8)
    ax.set_yscale('log')
    ax.set_xlim(-0.5, len(channels) - 0.5)

    ax.set_title('RMSE (log scale) of each variable comparison between OneScience and Official', fontsize=14, fontweight='bold', pad=10)

    # 样式
    ax.legend(frameon=False, loc='upper right', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    config_file_path = "../conf/config.yaml"
    cfg = YParams(config_file_path, "model")
    ## DataLoader init
    cfg_data = YParams(config_file_path, "datapipe")
    channels = cfg_data.dataset.channels
    datapath = cfg_data.dataset.data_dir

    h5_files = sorted(glob.glob(os.path.join(datapath, "data", "*.h5")))
    with h5py.File(h5_files[0], "r") as f:
        ds = f["fields"]
        variables = [v.decode() if isinstance(v, bytes) else v for v in ds.attrs["variables"]]
    channel_indices = [variables.index(v) for v in cfg_data.dataset.channels]
    show_compare(channel_indices)
    plot_rmse_comparison(channel_indices, filename='./result/rmse_comparison.png')
    date = f"{cfg_data.dataset.test_time[0]}010106"
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
    
    
