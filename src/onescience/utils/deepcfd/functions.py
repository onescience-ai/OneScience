import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
import os
import torch

def compute_metrics(pred, target, reduction='mean'):
    """
    计算 DeepCFD 预测结果的评估指标。
    
    Args:
        pred (torch.Tensor): 模型预测值 [B, C, H, W]
        target (torch.Tensor): 真实值 [B, C, H, W]
        reduction (str): 'mean' (返回标量) 或 'none' (返回每个样本的误差)
        
    Returns:
        dict: 包含各个物理分量和整体的误差指标字典
    """
    assert pred.shape == target.shape, "Prediction and Target shapes must match."
    
    metrics = {}
    
    # 物理量名称映射 (假设通道顺序为: 0->Ux, 1->Uy, 2->p)
    channel_names = ['Ux', 'Uy', 'p']
    num_channels = pred.shape[1]
    
    # --- 计算分量误差 (Component-wise) ---
    for i in range(num_channels):
        name = channel_names[i] if i < len(channel_names) else f"Ch{i}"
        
        diff = pred[:, i, ...] - target[:, i, ...]
        
        # MSE
        mse = torch.mean(diff ** 2).item()
        metrics[f'MSE_{name}'] = mse
        
        # MAE
        mae = torch.mean(torch.abs(diff)).item()
        metrics[f'MAE_{name}'] = mae
        
        # RMSE
        metrics[f'RMSE_{name}'] = np.sqrt(mse)

    diff_global = pred - target
    metrics['MSE_Global'] = torch.mean(diff_global ** 2).item()
    metrics['MAE_Global'] = torch.mean(torch.abs(diff_global)).item()

    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    
    diff_norm = torch.norm(pred_flat - target_flat, p=2, dim=1)
    target_norm = torch.norm(target_flat, p=2, dim=1)
    
    # 防止除以零，加上一个极小值
    rel_l2 = diff_norm / (target_norm + 1e-8)
    metrics['Rel_L2'] = torch.mean(rel_l2).item()

    return metrics
    
def split_tensors(*tensors, ratio):
    assert len(tensors) > 0
    split1, split2 = [], []
    count = len(tensors[0])
    for tensor in tensors:
        assert len(tensor) == count
        split1.append(tensor[:int(len(tensor) * ratio)])
        split2.append(tensor[int(len(tensor) * ratio):])
    if len(tensors) == 1:
        split1, split2 = split1[0], split2[0]
    return split1, split2

def initialize(model, gain=1, std=0.02):
    for module in model.modules():
        if type(module) in [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d]:
            nn.init.xavier_normal_(module.weight, gain)
            if module.bias is not None:
                nn.init.normal_(module.bias, 0, std)

def loss_func(output, target, weights):
    lossu = ((output[:, 0] - target[:, 0]) ** 2)
    lossv = ((output[:, 1] - target[:, 1]) ** 2)
    lossp = torch.abs((output[:, 2] - target[:, 2]))
    
    loss_stack = torch.stack([lossu, lossv, lossp], dim=1)
    weighted_loss = loss_stack / weights
    return torch.sum(weighted_loss)


def visualize(sample_y, out_y, error, s, save_dir="result"):
    # 创建 result 目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)


    minu = np.min(sample_y[s, 0, :, :])
    maxu = np.max(sample_y[s, 0, :, :])

    minv = np.min(sample_y[s, 1, :, :])
    maxv = np.max(sample_y[s, 1, :, :])

    minp = np.min(sample_y[s, 2, :, :])
    maxp = np.max(sample_y[s, 2, :, :])

    mineu = np.min(error[s, 0, :, :])
    maxeu = np.max(error[s, 0, :, :])

    minev = np.min(error[s, 1, :, :])
    maxev = np.max(error[s, 1, :, :])

    minep = np.min(error[s, 2, :, :])
    maxep = np.max(error[s, 2, :, :])

    nx = sample_y.shape[2]
    ny = sample_y.shape[3]

    plot_options = {'cmap': 'jet', 'origin': 'lower', 'extent': [0,nx,0,ny]}

    plt.figure()
    fig = plt.gcf()
    fig.set_size_inches(12, 8)

    plt.subplot(3, 3, 1)
    plt.title('CFD', fontsize=5)
    plt.imshow(np.transpose(sample_y[s, 0, :, :]), vmin = minu, vmax = maxu, **plot_options)
    plt.colorbar(orientation='horizontal')
    plt.ylabel('Ux', fontsize=5)

    plt.subplot(3, 3, 2)
    plt.title('CNN', fontsize=5)
    plt.imshow(np.transpose(out_y[s, 0, :, :]), vmin = minu, vmax =maxu, **plot_options)
    plt.colorbar(orientation='horizontal')

    plt.subplot(3, 3, 3)
    plt.title('Error', fontsize=5)
    plt.imshow(np.transpose(error[s, 0, :, :]), vmin = mineu, vmax = maxeu, **plot_options)
    plt.colorbar(orientation='horizontal')

    plt.subplot(3, 3, 4)
    plt.imshow(np.transpose(sample_y[s, 1, :, :]), vmin = minv, vmax = maxv, **plot_options)
    plt.colorbar(orientation='horizontal')
    plt.ylabel('Uy', fontsize=5)

    plt.subplot(3, 3, 5)
    plt.imshow(np.transpose(out_y[s, 1, :, :]), vmin = minv, vmax = maxv, **plot_options)
    plt.colorbar(orientation='horizontal')

    plt.subplot(3, 3, 6)
    plt.imshow(np.transpose(error[s, 1, :, :]), vmin = minev, vmax = maxev, **plot_options)
    plt.colorbar(orientation='horizontal')

    plt.subplot(3, 3, 7)
    plt.imshow(np.transpose(sample_y[s, 2, :, :]), vmin = minp, vmax = maxp, **plot_options)
    plt.colorbar(orientation='horizontal')
    plt.ylabel('p', fontsize=5)

    plt.subplot(3, 3, 8)
    plt.imshow(np.transpose(out_y[s, 2, :, :]), vmin = minp, vmax = maxp, **plot_options)
    plt.colorbar(orientation='horizontal')

    plt.subplot(3, 3, 9)
    plt.imshow(np.transpose(error[s, 2, :, :]), vmin = minep, vmax = maxep, **plot_options)
    plt.colorbar(orientation='horizontal')

    plt.tight_layout()

    save_path = os.path.join(save_dir, f"visualization_{s}.png")
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()
