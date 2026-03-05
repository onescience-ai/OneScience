import math
import torch
import torch.nn as nn
from einops import rearrange
import numpy as np

def unified_pos_embedding(shapelist, ref, batchsize=1, device='cuda'):
    """
    计算统一位置编码 (Unified Positional Embedding)。

    该函数在 [0, 1] 的归一化坐标空间内，计算输入网格（由 shapelist 定义）中每个点与参考网格（由 ref 定义）中每个点之间的欧几里得距离。
    该函数支持 1D、2D 和 3D 空间。它通常用于构建基于距离的相对位置编码或注意力偏置，将不同分辨率的物理网格映射到一组固定分辨率的参考锚点上。

    Args:
        shapelist (list[int]): 输入网格的形状列表。
            - 1D: [L]
            - 2D: [H, W]
            - 3D: [D, H, W]
        ref (int): 参考网格在每个维度上的分辨率。
            - 1D: 参考点数量为 ref
            - 2D: 参考点数量为 ref * ref
            - 3D: 参考点数量为 ref * ref * ref
        batchsize (int, optional): 批次大小。默认值: 1。
        device (str or torch.device, optional): 计算设备。默认值: 'cuda'。

    形状:
        输出: (B, N_input, N_ref)
            - B 为 batchsize。
            - N_input 为输入网格的总点数（即 prod(shapelist)）。
            - N_ref 为参考网格的总点数（即 ref ** len(shapelist)）。

    Example:
        >>> # 2D 示例: 输入网格 32x32, 参考网格 4x4
        >>> pos_embed = unified_pos_embedding([32, 32], ref=4, batchsize=2)
        >>> # 输入点数 N = 32*32 = 1024
        >>> # 参考点数 M = 4*4 = 16
        >>> pos_embed.shape
        torch.Size([2, 1024, 16])

        >>> # 1D 示例: 序列长度 100, 参考点 10
        >>> pos_embed_1d = unified_pos_embedding([100], ref=10, batchsize=1)
        >>> pos_embed_1d.shape
        torch.Size([1, 100, 10])
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    if len(shapelist) == 1:
        size_x = shapelist[0]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        grid = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1]).to(device)  # B N 1
        gridx = torch.tensor(np.linspace(0, 1, ref), dtype=torch.float)
        grid_ref = gridx.reshape(1, ref, 1).repeat([batchsize, 1, 1]).to(device)  # B N 1
        pos = torch.sqrt(torch.sum((grid[:, :, None, :] - grid_ref[:, None, :, :]) ** 2, dim=-1)). \
            reshape(batchsize, size_x, ref).contiguous()
    if len(shapelist) == 2:
        size_x, size_y = shapelist[0], shapelist[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        grid = torch.cat((gridx, gridy), dim=-1).to(device)  # B H W 2

        gridx = torch.tensor(np.linspace(0, 1, ref), dtype=torch.float)
        gridx = gridx.reshape(1, ref, 1, 1).repeat([batchsize, 1, ref, 1])
        gridy = torch.tensor(np.linspace(0, 1, ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, ref, 1).repeat([batchsize, ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy), dim=-1).to(device)  # B H W 8 8 2

        pos = torch.sqrt(torch.sum((grid[:, :, :, None, None, :] - grid_ref[:, None, None, :, :, :]) ** 2, dim=-1)). \
            reshape(batchsize, size_x * size_y, ref * ref).contiguous()
    if len(shapelist) == 3:
        size_x, size_y, size_z = shapelist[0], shapelist[1], shapelist[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        grid = torch.cat((gridx, gridy, gridz), dim=-1).to(device)  # B H W D 3

        gridx = torch.tensor(np.linspace(0, 1, ref), dtype=torch.float)
        gridx = gridx.reshape(1, ref, 1, 1, 1).repeat([batchsize, 1, ref, ref, 1])
        gridy = torch.tensor(np.linspace(0, 1, ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, ref, 1, 1).repeat([batchsize, ref, 1, ref, 1])
        gridz = torch.tensor(np.linspace(0, 1, ref), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, ref, 1).repeat([batchsize, ref, ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy, gridz), dim=-1).to(device)  # B 4 4 4 3

        pos = torch.sqrt(
            torch.sum((grid[:, :, :, :, None, None, None, :] - grid_ref[:, None, None, None, :, :, :, :]) ** 2,
                      dim=-1)). \
            reshape(batchsize, size_x * size_y * size_z, ref * ref * ref).contiguous()
    return pos