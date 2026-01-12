import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
import h5py
from tqdm import tqdm
from onescience.utils.fcn.YParams import YParams
from matplotlib import rcParams

# rcParams['font.family'] = 'serif'
# rcParams['font.serif'] = ['DejaVu Serif']
rcParams['mathtext.fontset'] = 'stix'
rcParams['axes.linewidth'] = 0.9
rcParams['xtick.major.width'] = 0.9
rcParams['ytick.major.width'] = 0.9


def get_metadata(cfg):
    meta_path = os.path.join(cfg.data_dir, 'metadata.json')
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    variables = metadata['variables']
    channel_indices = [variables.index(v) for v in cfg.channels]

    total_files = [f for f in os.listdir('./result/output/') if f.endswith('.npy')]
    total_files.sort()
    return total_files, channel_indices


def get_rmse(total_files, channel_indices):
    channel_rmse = np.zeros(len(channel_indices))
    if not os.path.exists('result/rmse.npy'):
        for file in tqdm(total_files, unit="files"):
            with h5py.File(f'{cfg_data.dataset.data_dir}/data/{file[:4]}/{file[:-4]}.h5', "r") as f:
                label = f["fields"][:]  # [N, H, W]
                label = label[channel_indices]
            pred = np.load(f'result/output/{file}').squeeze()
            channel_rmse += np.sqrt(np.mean((label - pred) ** 2, axis=(1, 2)))
        channel_rmse /= len(total_files)
        np.save('result/rmse.npy', channel_rmse)
    else:
        channel_rmse = np.load('result/rmse.npy')
    
    for i in range(len(channel_indices)):
        print(f"📂 Channel: {cfg_data.dataset.channels[i]} RMSE: {channel_rmse[i]: .4f},")
    
    channels = [cfg_data.dataset.channels[i] for i in range(len(channel_indices))]
    w = 24  # 最长 channel 名宽度
    
    # 表头
    print(f"┌{'─' * (w + 2)}┬{'─' * 14}┐")
    print(f"│ {'Channel':<{w}} │ {'OneScience':>12} |")
    print(f"├{'─' * (w + 2)}┼{'─' * 14}┤")
    # 数据行
    for i, ch in enumerate(channels):
        print(f"│ {ch:<{w}} │ {channel_rmse[i]:>12.4f} |")
    print(f"├{'─' * (w + 2)}┼{'─' * 14}┤")

    print(f"✅ Avg RMSE is : {np.mean(channel_rmse): .4f}")


def plot(label, pred, var, filename):
    # 基础设置
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 坐标轴标签
    xtick_labels = ['180°W', '90°W', '0°', '90°E', '180°E']
    ytick_labels = ['90°S', '45°S', '0°', '45°N', '90°N']
    xticks = np.linspace(0, label.shape[-1] - 1, 5)
    yticks = np.linspace(0, label.shape[-2] - 1, 5)
    
    # 计算统一色条范围
    vmin = min(label.min(), pred.min())
    vmax = max(label.max(), pred.max())
    
    # 计算差异和 RMSE
    diff = label - pred
    rmse = np.sqrt(np.mean(diff ** 2))
    diff_abs_max = np.abs(diff).max()
    
    # 绘图配置
    plot_configs = [
        {'data': label, 'title': 'Truth', 'cmap': 'viridis', 'vmin': vmin, 'vmax': vmax},
        {'data': pred,  'title': 'Prediction', 'cmap': 'viridis', 'vmin': vmin, 'vmax': vmax},
        {'data': diff,  'title': f'Difference (RMSE={rmse:.2f})', 'cmap': 'RdBu_r', 'vmin': -diff_abs_max, 'vmax': diff_abs_max},
    ]
    
    # 统一绘制
    for ax, cfg in zip(axes, plot_configs):
        im = ax.imshow(cfg['data'], cmap=cfg['cmap'], vmin=cfg['vmin'], vmax=cfg['vmax'])
        ax.set_title(cfg['title'], fontsize=12, pad=4)
        ax.set_xlabel('Longitude')  # 继续增大
        ax.set_ylabel('Latitude')
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels)
        plt.colorbar(im, ax=ax, orientation='horizontal')# , pad=0.04, aspect=30, shrink=0.85
    
    # 总标题 - 降低位置
    fig.suptitle(var, fontsize=14, fontweight='bold', y=0.98)
    
    # plt.tight_layout(rect=[0, 0.08, 1, 0.95])  # 底部也留空间
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_loss(train_loss, valid_loss):

    mask = ~(np.isnan(train_loss) | np.isnan(valid_loss))
    train_loss = train_loss[mask]
    valid_loss = valid_loss[mask]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    # 配置
    colors = {'train': '#2563EB', 'valid': '#EA580C'}
    epochs = np.arange(1, len(train_loss) + 1)
    
    # 绑定曲线
    ax.plot(epochs, train_loss, color=colors['train'], linewidth=1.5, label='Train')
    ax.plot(epochs, valid_loss, color=colors['valid'], linewidth=1.5, label='Valid', linestyle='--')
    # 标注最小值
    min_idx = np.argmin(valid_loss)
    ax.scatter(epochs[min_idx], valid_loss[min_idx], 
               color=colors['valid'], s=40, zorder=5, edgecolors='white')
    ax.annotate(f'Best: {valid_loss[min_idx]:.3f}', 
                xy=(epochs[min_idx], valid_loss[min_idx]),
                xytext=(10, 10), textcoords='offset points', fontsize=8, color=colors['valid'],
                arrowprops=dict(arrowstyle='-', color=colors['valid'], lw=0.5))
    
    # 坐标轴
    ax.set(xlabel='Epoch', ylabel='Loss', xlim=(0, len(train_loss) + 1))
    
    # 样式
    ax.legend(frameon=False, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('./result/loss.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    current_path = os.getcwd()
    sys.path.append(current_path)
    config_file_path = os.path.join(current_path, 'conf/config.yaml')
    cfg = YParams(config_file_path, 'model')
    cfg_data = YParams(config_file_path, "datapipe")

    train_loss = np.load('./data/checkpoints/trloss.npy')
    valid_loss = np.load('./data/checkpoints/valoss.npy')
    plot_loss(train_loss, valid_loss)
    total_files, channel_indices = get_metadata(cfg_data.dataset)

    # Load data
    # Compute RMSE per channel and total
    get_rmse(total_files, channel_indices)

    ##### You can choose the date to plot (must exist in ./result/output/)#####
    total_files = ['2020100100']
    channel_index = [cfg_data.dataset.channels.index(v) for v in ['2m_temperature', 'geopotential_500', 'temperature_500']] # '2m_temperature', 'geopotential_500', 'temperature_500'

    ##### Or use random index to plot #####
    # np.random.seed(42) # use a fix seed ensure to get same result
    # sample_index = np.random.choice(len(total_files), 3, replace=False)
    # channel_index = np.random.choice(len(cfg_data.dataset.channels), 3, replace=False)
    # selected_files = [total_files[int(i)] for i in sample_index]

    
    selected_var = [cfg_data.dataset.channels[int(i)] for i in channel_index]
    print(f"seleted date: {total_files}")
    print(f"selected channels: {selected_var}")
    for file in total_files:
        with h5py.File(f'{cfg_data.dataset.data_dir}/data/{file[:4]}/{file}.h5', "r") as f:
            label = f["fields"][:]  # [N, H, W]
            label = label[channel_indices]
        pred = np.load(f'result/output/{file}.npy').squeeze()
        for i in range(len(selected_var)):
            filename = f'./result/{file}_{selected_var[i]}.png'
            plot(label[channel_index[i]], pred[channel_index[i]], selected_var[i], filename)
            print(f'✅plot {filename}')