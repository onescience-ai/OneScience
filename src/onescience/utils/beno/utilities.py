import torch
import numpy as np
import scipy.io
import h5py
import sklearn.metrics
from torch_geometric.data import Data
import torch.nn as nn
from scipy.ndimage import gaussian_filter
from torch_geometric.nn import GCNConv
import pdb
import matplotlib.pyplot as plt


def compute_boundary_gradient(bc_all, sol_all, coord_all, resolution):
    """
    从边界值及网格内部值计算法向梯度（Neumann 边界条件）

    Args:
        bc_all: 边界点信息 [n_samples, n_boundary_points, 3]，包含边界点的 (x, y, 值)
        sol_all: 全部网格解 [n_samples, n_points_per_sample]，包含内部点的解值
        coord_all: 网格内部所有点的坐标 [n_samples, n_points_per_sample, 2]
        resolution: 网格的分辨率，例如 32x32

    Returns:
        bc_gradient: 边界点的法向梯度，形状为 [n_samples, n_boundary_points]
    """
    n_samples, n_boundary_points, _ = bc_all.shape
    bc_gradient = np.zeros((n_samples, n_boundary_points))  # 输出法向梯度

    dx = 1 / (resolution - 1)  # 假设网格均匀分布在 [0, 1] 区间

    for sample_idx in range(n_samples):
        for i in range(n_boundary_points):
            # 当前边界点的 (x, y) 坐标及其值
            bx, by, b_value = bc_all[sample_idx, i]

            # 找到该边界点在 coord_all 中的索引
            boundary_idx = np.where(
                (coord_all[sample_idx][:, 0] == bx)
                & (coord_all[sample_idx][:, 1] == by)
            )[0][0]

            # 找到该边界点相邻的内部点索引
            neighbors = []
            if boundary_idx % resolution != 0:  # 左边不是边界
                neighbors.append(boundary_idx - 1)
            if boundary_idx % resolution != resolution - 1:  # 右边不是边界
                neighbors.append(boundary_idx + 1)
            if boundary_idx >= resolution:  # 上方不是边界
                neighbors.append(boundary_idx - resolution)
            if boundary_idx < resolution * (resolution - 1):  # 下方不是边界
                neighbors.append(boundary_idx + resolution)

            # 计算法向量（假设均匀网格，单位法向量由邻点和当前点的坐标推导）
            normal_vector = np.array(
                [
                    bx - coord_all[sample_idx][neighbors, 0].mean(),
                    by - coord_all[sample_idx][neighbors, 1].mean(),
                ]
            )
            normal_vector = normal_vector / np.linalg.norm(normal_vector)

            # 计算梯度（解的方向导数）
            neighbor_values = sol_all[sample_idx, neighbors]
            bc_gradient[sample_idx, i] = (
                np.dot(neighbor_values - b_value, normal_vector) / dx
            )

    return bc_gradient


import os  # 需要导入os模块来处理目录操作


def plot_data(
    predict_term,
    true_term,
    forcing_term,
    forcing_mask,
    grid_info,
    resolution,
    num_samples=3,
    interpolation="bilinear",
    save_path=None,
):
    """
    绘制源项 f 和解项 u 的云图，只绘制内部点和边界点，并支持保存到文件。

    Args:
        predict_term: 预测解项 (nsamples, resolution, resolution)
        true_term: 真实解项 (nsamples, resolution, resolution)
        forcing_term: 源项 (nsamples, resolution, resolution)
        forcing_mask: 标志点 mask (nsamples, npoints)
        num_samples: 要绘制的样本数量
        interpolation: 图像插值方法
        save_path: 如果提供路径，则保存图片到该路径 (str)
    """
    predict_term = predict_term.reshape(-1, resolution, resolution)
    true_term = true_term.reshape(-1, resolution, resolution)
    forcing_term = forcing_term.reshape(-1, resolution, resolution)
    forcing_mask = forcing_mask.reshape(-1, resolution, resolution)
    grid_info = grid_info.reshape(-1, resolution, resolution, 2)
    # 获取 true_term 的第一个维度的大小
    num_total_samples = true_term.shape[0]
    # 从第一个维度中随机选取 num_samples 个样本
    if num_samples > num_total_samples:
        raise ValueError(
            f"num_samples ({num_samples}) cannot be greater than the total number of samples ({num_total_samples})."
        )
    sample_indices = np.random.choice(num_total_samples, num_samples, replace=False)

    fig, axes = plt.subplots(4, num_samples, figsize=(4 * num_samples, 8))

    for idx, i in enumerate(sample_indices):
        # 第 i 个样本的源项 f 和解项 u
        f = forcing_term[i]
        p_u = predict_term[i]
        t_u = true_term[i]
        mask = forcing_mask[i]
        error = np.abs(t_u - p_u)
        # 创建掩码，只显示内部点
        internal_mask = mask == 0  # 内部点

        grid_x = grid_info[i, :, :, 0]  # 第 i 个样本的 x 坐标
        grid_y = grid_info[i, :, :, 1]  # 第 i 个样本的 y 坐标

        # 应用掩码，只显示内部点
        f_masked = np.where(internal_mask, f, np.nan)
        p_u_masked = np.where(internal_mask, p_u, np.nan)
        t_u_masked = np.where(internal_mask, t_u, np.nan)
        error_masked = np.where(internal_mask, error, np.nan)

        # 绘制源项 f
        ax_f = axes[0, idx]
        im_f = ax_f.imshow(
            f_masked,
            cmap="viridis",
            origin="lower",
            extent=[0, 1, 0, 1],
            interpolation=interpolation,
        )
        ax_f.set_title(f"(a) Forcing term $f$ (Sample {i+1})", fontsize=12)
        plt.colorbar(im_f, ax=ax_f, fraction=0.046, pad=0.04)

        # 绘制真实解项 u
        ax_t_u = axes[1, idx]
        im_t_u = ax_t_u.imshow(
            t_u_masked,
            cmap="viridis",
            origin="lower",
            extent=[0, 1, 0, 1],
            interpolation=interpolation,
        )
        ax_t_u.set_title(f"(b) True Solution term $u$ (Sample {i+1})", fontsize=12)
        plt.colorbar(im_t_u, ax=ax_t_u, fraction=0.046, pad=0.04)

        # 绘制预测解项 u
        ax_p_u = axes[2, idx]
        im_p_u = ax_p_u.imshow(
            p_u_masked,
            cmap="viridis",
            origin="lower",
            extent=[0, 1, 0, 1],
            interpolation=interpolation,
        )
        ax_p_u.set_title(f"(c) Predict Solution term $u$ (Sample {i+1})", fontsize=12)
        plt.colorbar(im_p_u, ax=ax_p_u, fraction=0.046, pad=0.04)

        # 绘制预测误差
        ax_error = axes[3, idx]
        im_error = ax_error.imshow(
            error_masked,
            cmap="viridis",
            origin="lower",
            extent=[0, 1, 0, 1],
            interpolation=interpolation,
        )
        ax_error.set_title(f"(d) Absolute Error (Sample {i+1})", fontsize=12)
        plt.colorbar(im_error, ax=ax_error, fraction=0.046, pad=0.04)

    # 布局调整
    plt.tight_layout()

    if save_path:
        # 检查目录是否存在，如果不存在则创建
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        # 保存图片到指定路径
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        # 显示图片
        plt.show()


class MeshGenerator(object):
    def __init__(self, real_space, mesh_size, attr_features=1, grid_input=np.array([])):
        super(MeshGenerator, self).__init__()

        self.d = len(real_space)  # 2
        # self.m = sample_size  #1000
        self.attr_features = attr_features

        assert len(mesh_size) == self.d

        if self.d == 1:  # 如果是一维网格，节点数 = 网格点数
            self.n = mesh_size[0]
            # self.grid = np.linspace(real_space[0][0], real_space[0][1], self.n).reshape((self.n, 1))
        else:
            self.n = 1
            grids = []
            for j in range(self.d):
                # grids.append(np.linspace(real_space[j][0], real_space[j][1], mesh_size[j]))
                # grids.append(np.linspace(real_space[j][0]+(0.5/mesh_size[j]), real_space[j][1]-(0.5/mesh_size[j]), mesh_size[j]))
                self.n *= mesh_size[j]  # 对于多维网格，节点数 = 各维度网格点数的乘积

        self.idx = np.array(range(self.n))
        self.grid = grid_input
        self.grid_sample = self.grid

    def sample(self, idx):
        self.idx = torch.tensor(idx)
        self.grid_sample = self.grid[self.idx]
        return self.idx

    def get_grid(self):  # 将当前采样的网格（grid_sample）返回为 PyTorch 张量
        return torch.tensor(self.grid_sample, dtype=torch.float)

    def deduplicate_rows(tensor):  # 去除输入张量 tensor 中的重复行
        unique_rows = []
        seen_rows = set()
        for row in tensor:
            row_tuple = tuple(row.tolist())
            if row_tuple not in seen_rows:
                unique_rows.append(row)
                seen_rows.add(row_tuple)
        deduplicated_tensor = torch.stack(unique_rows)
        return deduplicated_tensor

    def ball_connectivity(
        self, is_forward=False, ns=10, tri_edge=None
    ):  # 根据输入的网格节点计算边索引，构造图的连接关系。
        self.pwd = sklearn.metrics.pairwise_distances(self.grid_sample)
        tri_edge = tri_edge.T

        edge_index_1 = np.array([])
        edge_index_2 = np.array([])
        for i in range(self.grid_sample.shape[0]):
            edge_index_1 = np.append(edge_index_1, np.array([i]).repeat(ns + 1))
            edge_index_2 = np.append(edge_index_2, np.argsort(self.pwd[i])[: ns + 1])
        self.edge_index = np.vstack([edge_index_1, edge_index_2])
        self.edge_index = np.concatenate([self.edge_index, tri_edge], -1)

        self.edge_index = torch.tensor(self.edge_index)
        self.edge_index = torch.cat([self.edge_index, self.edge_index.flip(0)], dim=1)
        self.edge_index = MeshGenerator.deduplicate_rows(self.edge_index.T).T
        self.n_edges = self.edge_index.shape[1]
        if is_forward:
            print(self.edge_index.shape)
            self.edge_index = self.edge_index[
                :, self.edge_index[0] >= self.edge_index[1]
            ]
            print(self.edge_index.shape)
            self.n_edges = self.edge_index.shape[1]

        return torch.tensor(self.edge_index, dtype=torch.long)

    def attributes(
        self, theta=None
    ):  # 构造边特征矩阵 edge_attr 包括：1.几何特性（如边长） 2.起点和终点的属性 3.起点和终点的坐标
        # pwd = sklearn.metrics.pairwise_distances(self.grid_sample)
        theta = theta[self.idx]
        edge_attr = np.zeros((self.n_edges, 2 * self.d + 2 * self.attr_features + 1))
        self.edge_index = torch.tensor(self.edge_index).to(torch.int64)

        for p in range(self.n_edges):
            edge_attr[p, 6:7] = self.pwd[self.edge_index[0][p]][self.edge_index[1][p]]
        edge_attr[:, 4:5] = theta[self.edge_index[0]].view(-1, self.attr_features)
        edge_attr[:, 5:6] = theta[self.edge_index[1]].view(-1, self.attr_features)
        edge_attr[:, 0:4] = self.grid_sample[self.edge_index.T].reshape(
            (self.n_edges, -1)
        )

        return torch.tensor(edge_attr, dtype=torch.float)


# normalization, Gaussian
class GaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    # def cuda(self):
    #     self.mean = self.mean.cuda()
    #     self.std = self.std.cuda()

    # def cpu(self):
    #     self.mean = self.mean.cpu()
    #     self.std = self.std.cpu()
    def cuda(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=False, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(
            x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1
        )

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]  # x.size()=[1,num_indomain]

        diff_norms = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1
        )  # pred-gd 求L2范数
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


class RandomMeshGenerator(object):
    def __init__(self, real_space, mesh_size, sample_size):
        super(RandomMeshGenerator, self).__init__()

        self.d = len(real_space)
        self.m = sample_size

        assert len(mesh_size) == self.d

        if self.d == 1:
            self.n = mesh_size[0]
            self.grid = np.linspace(real_space[0][0], real_space[0][1], self.n).reshape(
                (self.n, 1)
            )
        else:
            self.n = 1
            grids = []
            for j in range(self.d):
                grids.append(
                    np.linspace(real_space[j][0], real_space[j][1], mesh_size[j])
                )
                self.n *= mesh_size[j]

            self.grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T

        if self.m > self.n:
            self.m = self.n

        self.idx = np.array(range(self.n))
        self.grid_sample = self.grid

    def sample(self):
        perm = torch.randperm(self.n)
        self.idx = perm[: self.m]
        self.grid_sample = self.grid[self.idx]
        return self.idx

    def get_grid(self):
        return torch.tensor(self.grid_sample, dtype=torch.float)

    def ball_connectivity(self, r):
        pwd = sklearn.metrics.pairwise_distances(self.grid_sample)
        self.edge_index = np.vstack(np.where(pwd <= r))
        self.n_edges = self.edge_index.shape[1]

        return torch.tensor(self.edge_index, dtype=torch.long)

    def gaussian_connectivity(self, sigma):
        pwd = sklearn.metrics.pairwise_distances(self.grid_sample)
        rbf = np.exp(-(pwd**2) / sigma**2)
        sample = np.random.binomial(1, rbf)
        self.edge_index = np.vstack(np.where(sample))
        self.n_edges = self.edge_index.shape[1]
        return torch.tensor(self.edge_index, dtype=torch.long)

    def attributes(self, f=None, theta=None):
        if f is None:
            if theta is None:
                edge_attr = self.grid[self.edge_index.T].reshape((self.n_edges, -1))
            else:
                theta = theta[self.idx]
                edge_attr = np.zeros((self.n_edges, 3 * self.d))
                edge_attr[:, 0 : 2 * self.d] = self.grid_sample[
                    self.edge_index.T
                ].reshape((self.n_edges, -1))
                edge_attr[:, 2 * self.d] = theta[self.edge_index[0]]
                edge_attr[:, 2 * self.d + 1] = theta[self.edge_index[1]]
        else:
            xy = self.grid_sample[self.edge_index.T].reshape((self.n_edges, -1))
            if theta is None:
                edge_attr = f(xy[:, 0 : self.d], xy[:, self.d :])
            else:
                theta = theta[self.idx]
                edge_attr = f(
                    xy[:, 0 : self.d],
                    xy[:, self.d :],
                    theta[self.edge_index[0]],
                    theta[self.edge_index[1]],
                )

        return torch.tensor(edge_attr, dtype=torch.float)
