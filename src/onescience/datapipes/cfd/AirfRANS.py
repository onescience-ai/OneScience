import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from onescience.distributed.manager import DistributedManager

import numpy as np
import pyvista as pv
import torch
import torch_geometric.nn as nng
from torch_geometric.data import Data
from tqdm import tqdm

from onescience.datapipes.core import BaseDataset
from onescience.utils.transolver.reorganize import reorganize 

from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader as PyGDataLoader # 使用 PyG 的 DataLoader

def cell_sampling_2d(cell_points, cell_attr=None):
    """
    通过平行四边形采样和基于重心坐标的三角形插值，在二维单元中采样点。顶点必须按特定顺序排列。

    Args:
        cell_points (array): 二维单元的顶点。形状为 (N, 4)，表示 N 个具有 4 个顶点的单元。
        cell_attr (array, optional): 二维单元顶点的特征。形状为 (N, 4, k)。
            如果形状为 (N, 4)，将自动调整为 (N, 4, 1)。默认为 ``None``。
    """
    # 通过单元三角剖分和平行四边形采样进行采样
    v0, v1 = (
        cell_points[:, 1] - cell_points[:, 0],
        cell_points[:, 3] - cell_points[:, 0],
    )
    v2, v3 = (
        cell_points[:, 3] - cell_points[:, 2],
        cell_points[:, 1] - cell_points[:, 2],
    )
    a0, a1 = np.abs(
        np.linalg.det(np.hstack([v0[:, :2], v1[:, :2]]).reshape(-1, 2, 2))
    ), np.abs(np.linalg.det(np.hstack([v2[:, :2], v3[:, :2]]).reshape(-1, 2, 2)))
    p = a0 / (a0 + a1)
    index_triangle = np.random.binomial(1, p)[:, None]
    u = np.random.uniform(size=(len(p), 2))
    sampled_point = index_triangle * (u[:, 0:1] * v0 + u[:, 1:2] * v1) + (
        1 - index_triangle
    ) * (u[:, 0:1] * v2 + u[:, 1:2] * v3)
    sampled_point_mirror = index_triangle * (
        (1 - u[:, 0:1]) * v0 + (1 - u[:, 1:2]) * v1
    ) + (1 - index_triangle) * ((1 - u[:, 0:1]) * v2 + (1 - u[:, 1:2]) * v3)
    reflex = u.sum(axis=1) > 1
    sampled_point[reflex] = sampled_point_mirror[reflex]

    # 基于重心坐标在三角形上进行插值
    if cell_attr is not None:
        t0, t1, t2 = (
            np.zeros_like(v0),
            index_triangle * v0 + (1 - index_triangle) * v2,
            index_triangle * v1 + (1 - index_triangle) * v3,
        )
        w = (t1[:, 1] - t2[:, 1]) * (t0[:, 0] - t2[:, 0]) + (t2[:, 0] - t1[:, 0]) * (
            t0[:, 1] - t2[:, 1]
        )
        w0 = (t1[:, 1] - t2[:, 1]) * (sampled_point[:, 0] - t2[:, 0]) + (
            t2[:, 0] - t1[:, 0]
        ) * (sampled_point[:, 1] - t2[:, 1])
        w1 = (t2[:, 1] - t0[:, 1]) * (sampled_point[:, 0] - t2[:, 0]) + (
            t0[:, 0] - t2[:, 0]
        ) * (sampled_point[:, 1] - t2[:, 1])
        w0, w1 = w0 / w, w1 / w
        w2 = 1 - w0 - w1

        if len(cell_attr.shape) == 2:
            cell_attr = cell_attr[:, :, None]
        attr0 = (
            index_triangle * cell_attr[:, 0] + (1 - index_triangle) * cell_attr[:, 2]
        )
        attr1 = (
            index_triangle * cell_attr[:, 1] + (1 - index_triangle) * cell_attr[:, 1]
        )
        attr2 = (
            index_triangle * cell_attr[:, 3] + (1 - index_triangle) * cell_attr[:, 3]
        )
        sampled_attr = w0[:, None] * attr0 + w1[:, None] * attr1 + w2[:, None] * attr2

    sampled_point += (
        index_triangle * cell_points[:, 0] + (1 - index_triangle) * cell_points[:, 2]
    )

    return (
        np.hstack([sampled_point[:, :2], sampled_attr])
        if cell_attr is not None
        else sampled_point[:, :2]
    )


def cell_sampling_1d(line_points, line_attr=None):
    """
    通过线性采样和插值在如一维单元中采样点。

    Args:
        line_points (array): 一维单元的边。形状为 (N, 2)，表示 N 个具有 2 条边的单元。
        line_attr (array, optional): 一维单元边的特征。形状为 (N, 2, k)。
            如果形状为 (N, 2)，将自动调整为 (N, 2, 1)。默认为 ``None``。
    """
    # 线性采样
    u = np.random.uniform(size=(len(line_points), 1))
    sampled_point = u * line_points[:, 0] + (1 - u) * line_points[:, 1]

    # 线性插值
    if line_attr is not None:
        if len(line_attr.shape) == 2:
            line_attr = line_attr[:, :, None]
        sampled_attr = u * line_attr[:, 0] + (1 - u) * line_attr[:, 1]

    return (
        np.hstack([sampled_point[:, :2], sampled_attr])
        if line_attr is not None
        else sampled_point[:, :2]
    )


class AirfRANSDataset(BaseDataset):
    """
    AirfRANS (CFD 翼型) 数据集
    
    继承自 BaseDataset，用于处理 PyTorch Geometric 的 Data 对象。
    """
    
    # 覆盖元数据
    DOMAIN = "cfd"
    TASK = "regression"
    DATA_FORMATS = ["vtu", "vtp"]

    def __init__(self, config: Union[Dict[str, Any]], mode: str = 'train', coef_norm: Optional[Tuple] = None):
        """
        初始化 AirfRANS 数据集
        
        Parameters
        ----------
        config : Dict[str, Any]
            数据集配置
        mode : str, optional
            'train', 'val', 或 'test'
        coef_norm : tuple, optional
            (mean_in, std_in, mean_out, std_out) 归一化系数。
            如果为 'train' 且此项为 None，将尝试加载或计算。
            如果为 'val'/'test'，必须提供此项。
        """
        self.mode = mode
        self._provided_coef_norm = coef_norm
        self.data_list_names = []
        self.coef_norm = None
        self.dist = DistributedManager()
        
        super().__init__(config)
        if self.logger.hasHandlers():
            self.logger.handlers.clear()      
                
        # 初始化
        self._init_paths()
        self._load_metadata() # 加载或计算归一化统计量
        if self.dist.rank == 0:
            self.logger.info(f"[{self.mode}] AirfRANS dataset initialized.")
            self.logger.info(f"[{self.mode}] Found {len(self.data_list_names)} simulation files.")

    def _init_paths(self):
        """
        加载 manifest.json 并根据 mode 拆分数据集
        """
        super()._init_paths() 
        manifest_path = self.data_path / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest.json not found at: {self.data_path}")
            
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        # 从配置中获取要使用的 manifest 键
        task_name = self.config.data.splits.task # e.g., 'full'
        
        if self.mode == 'train':
            train_key = self.config.data.splits.train_name # e.g., 'full_train'
            full_set = manifest[train_key]
            # 根据配置比例划分验证集
            n_val = int(len(full_set) * self.config.data.splits.val_split_ratio)
            self.data_list_names = full_set[:-n_val]
        elif self.mode == 'val':
            train_key = self.config.data.splits.train_name
            full_set = manifest[train_key]
            n_val = int(len(full_set) * self.config.data.splits.val_split_ratio)
            self.data_list_names = full_set[-n_val:]
        elif self.mode == 'test':
            test_key = self.config.data.splits.test_name # e.g., 'full_test'
            self.data_list_names = manifest[test_key]
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def _load_metadata(self):
        """
        加载或计算归一化系数
        """
        stats_dir = Path(self.config.source.stats_dir)
        stats_dir.mkdir(parents=True, exist_ok=True)
        
        mean_in_path = stats_dir / "mean_in.npy"
        std_in_path = stats_dir / "std_in.npy"
        mean_out_path = stats_dir / "mean_out.npy"
        std_out_path = stats_dir / "std_out.npy"
        
        if self._provided_coef_norm:
            if self.dist.rank == 0:
                self.logger.debug(f"[{self.mode}] Using provided normalization coefficients.")
            self.coef_norm = self._provided_coef_norm
            return

        if mean_in_path.exists() and std_in_path.exists() and mean_out_path.exists() and std_out_path.exists():
            if self.dist.rank == 0:
                self.logger.info(f"[{self.mode}] Loading normalization stats from {stats_dir}")
            mean_in = np.load(mean_in_path)
            std_in = np.load(std_in_path)
            mean_out = np.load(mean_out_path)
            std_out = np.load(std_out_path)
            self.coef_norm = (mean_in, std_in, mean_out, std_out)
        elif self.mode == 'train':
            if self.dist.rank == 0:
                self.logger.warning(f"[{self.mode}] Stats not found. Calculating normalization stats on the fly...")
            self.coef_norm = self._calculate_normalization()
            # 保存统计数据
            np.save(mean_in_path, self.coef_norm[0])
            np.save(std_in_path, self.coef_norm[1])
            np.save(mean_out_path, self.coef_norm[2])
            np.save(std_out_path, self.coef_norm[3])
            if self.dist.rank == 0:
                self.logger.info(f"[{self.mode}] Saved normalization stats to {stats_dir}")
        else:
            raise FileNotFoundError(f"[{self.mode}] Normalization stats not found in {stats_dir}, and mode is not 'train'.")
            
    def _calculate_normalization(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        在训练集上计算归一化统计量 (均值和标准差)
        这会遍历整个训练集，可能较慢
        """
        if self.mode != 'train':
            raise RuntimeError("Normalization calculation should only be done on the training set.")

        mean_in = None
        mean_out = None
        std_in = None
        std_out = None
        old_length_in = 0
        old_length_out = 0

        # 第一轮遍历：计算均值
        self.logger.info("Calculating mean...")
        # 注意：这里直接使用 _load_single_simulation 加载数据
        pbar = tqdm(self.data_list_names, desc="Norm (pass 1/2)")
        for s in pbar:
            # 临时禁用采样以统计原始网格数据
            # 避免统计结果依赖于特定采样策略
            original_sample_strategy = self.config.data.sampling.sample_strategy
            if original_sample_strategy is not None:
                self.config.data.sampling.sample_strategy = None
                pbar.set_description(f"Norm (pass 1/2) [Temp force sample=None for stats]")
            
            _, init, target, _ = self._load_single_simulation(s)
            
            if original_sample_strategy is not None:
                self.config.data.sampling.sample_strategy = original_sample_strategy

            if mean_in is None:
                mean_in = init.mean(axis=0, dtype=np.double)
                mean_out = target.mean(axis=0, dtype=np.double)
                old_length_in = init.shape[0]
            else:
                new_length = old_length_in + init.shape[0]
                mean_in += (init.sum(axis=0, dtype=np.double) - init.shape[0] * mean_in) / new_length
                mean_out += (target.sum(axis=0, dtype=np.double) - init.shape[0] * mean_out) / new_length
                old_length_in = new_length

        mean_in = mean_in.astype(np.single)
        mean_out = mean_out.astype(np.single)

        # 第二轮遍历：计算标准差
        self.logger.info("Calculating std dev...")
        old_length_in = 0 # 重置计数
        pbar = tqdm(self.data_list_names, desc="Norm (pass 2/2)")
        for s in pbar:
            original_sample_strategy = self.config.data.sampling.sample_strategy
            if original_sample_strategy is not None:
                self.config.data.sampling.sample_strategy = None
                pbar.set_description(f"Norm (pass 2/2) [Temp force sample=None for stats]")

            _, init, target, _ = self._load_single_simulation(s)
            
            if original_sample_strategy is not None:
                self.config.data.sampling.sample_strategy = original_sample_strategy
            
            if std_in is None:
                old_length_in = init.shape[0] # 使用第一轮记录的长度
                std_in = ((init - mean_in) ** 2).sum(axis=0, dtype=np.double) / old_length_in
                std_out = ((target - mean_out) ** 2).sum(axis=0, dtype=np.double) / old_length_in
            else:
                new_length = old_length_in + init.shape[0]
                std_in += (((init - mean_in) ** 2).sum(axis=0, dtype=np.double) - init.shape[0] * std_in) / new_length
                std_out += (((target - mean_out) ** 2).sum(axis=0, dtype=np.double) - init.shape[0] * std_out) / new_length
                old_length_in = new_length

        std_in = np.sqrt(std_in).astype(np.single)
        std_out = np.sqrt(std_out).astype(np.single)

        return (mean_in, std_in, mean_out, std_out)
        
    def _load_single_simulation(self, s: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        加载单个模拟文件，返回 (pos, x, y, surf_bool)
        """
        internal_path = self.data_path / s / f"{s}_internal.vtu"
        aerofoil_path = self.data_path / s / f"{s}_aerofoil.vtp"
        
        internal = pv.read(internal_path)
        aerofoil = pv.read(aerofoil_path)
        internal = internal.compute_cell_sizes(length=False, volume=False)
    
        if self.config.data.crop:
            bounds = (*self.config.data.crop, 0, 1)
            internal = internal.clip_box(bounds=bounds, invert=False, crinkle=True)

        sample_strategy = self.config.data.sampling.sample_strategy
        
        Uinf, alpha = float(s.split("_")[2]), float(s.split("_")[3]) * np.pi / 180
        
        if sample_strategy:
            # --- 采样模式 (sample != None) ---
            n_boot = self.config.data.sampling.n_boot
            surf_ratio = self.config.data.sampling.surf_ratio
            n_boot_surf = int(n_boot * surf_ratio)

            if sample_strategy == "uniform":
                p = internal.cell_data["Area"] / internal.cell_data["Area"].sum()
                sampled_cell_indices = np.random.choice(internal.n_cells, size=n_boot, p=p)
                
                surf_p = aerofoil.cell_data["Length"] / aerofoil.cell_data["Length"].sum()
                sampled_line_indices = np.random.choice(aerofoil.n_cells, size=n_boot_surf, p=surf_p)
                
            elif sample_strategy == "mesh":
                sampled_cell_indices = np.random.choice(internal.n_cells, size=n_boot)
                sampled_line_indices = np.random.choice(aerofoil.n_cells, size=n_boot_surf)
            else:
                raise ValueError(f"Unknown sample strategy: {sample_strategy}")

            # 处理体积网格 (volume)
            cell_dict = internal.cells.reshape(-1, 5)[sampled_cell_indices, 1:]
            cell_points = internal.points[cell_dict]
            
            geom = -internal.point_data["implicit_distance"][cell_dict, None]
            u = (np.array([np.cos(alpha), np.sin(alpha)]) * Uinf).reshape(1, 2) * \
                np.ones_like(internal.point_data["U"][cell_dict, :1])
            normal = np.zeros_like(u)

            attr = np.concatenate([
                u,
                geom,
                normal,
                internal.point_data["U"][cell_dict, :2],
                internal.point_data["p"][cell_dict, None],
                internal.point_data["nut"][cell_dict, None],
            ], axis=-1)
            
            # 执行 2D 采样
            sampled_points = cell_sampling_2d(cell_points, attr)
            
            pos_vol = sampled_points[:, :2]
            # 拼接坐标以匹配 7 维特征
            init_vol_base = sampled_points[:, 2:7] 
            init_vol = np.concatenate([pos_vol, init_vol_base], axis=1) # 7 dims
            target_vol = sampled_points[:, 7:] # [v_x, v_y, p, nut] (4 dims)

            # 处理表面网格 (surface)
            line_dict = aerofoil.lines.reshape(-1, 3)[sampled_line_indices, 1:]
            line_points = aerofoil.points[line_dict]
            
            surf_geom = np.zeros_like(aerofoil.point_data["U"][line_dict, :1])
            surf_u = (np.array([np.cos(alpha), np.sin(alpha)]) * Uinf).reshape(1, 2) * \
                     np.ones_like(aerofoil.point_data["U"][line_dict, :1])
            surf_normal = -aerofoil.point_data["Normals"][line_dict, :2]

            surf_attr = np.concatenate([
                surf_u,
                surf_geom,
                surf_normal,
                aerofoil.point_data["U"][line_dict, :2],
                aerofoil.point_data["p"][line_dict, None],
                aerofoil.point_data["nut"][line_dict, None],
            ], axis=-1)
            
            # 执行 1D 采样
            surf_sampled_points = cell_sampling_1d(line_points, surf_attr)

            pos_surf = surf_sampled_points[:, :2]
            init_surf_base = surf_sampled_points[:, 2:7]
            init_surf = np.concatenate([pos_surf, init_surf_base], axis=1) # 7 dims
            target_surf = surf_sampled_points[:, 7:] # 4 dims

            # 合并体积和表面数据
            pos = np.concatenate([pos_vol, pos_surf], axis=0)
            x = np.concatenate([init_vol, init_surf], axis=0)
            y = np.concatenate([target_vol, target_surf], axis=0)
            surf_bool = np.concatenate([
                np.zeros(len(pos_vol), dtype=bool), 
                np.ones(len(pos_surf), dtype=bool)
            ], axis=0)
            
            return pos, x, y, surf_bool

        else:
            # --- 全量网格模式 (sample=None) ---
            surf_bool = internal.point_data["U"][:, 0] == 0
            geom = -internal.point_data["implicit_distance"][:, None]
            u = (np.array([np.cos(alpha), np.sin(alpha)]) * Uinf).reshape(1, 2) * \
                np.ones_like(internal.point_data["U"][:, :1])
            normal = np.zeros_like(u)
            
            # 重组表面点数据
            normal[surf_bool] = reorganize(
                aerofoil.points[:, :2],
                internal.points[surf_bool, :2],
                -aerofoil.point_data["Normals"][:, :2],
            )
            
            pos = internal.points[:, :2]
            init = np.concatenate([
                pos,
                u,
                geom,
                normal,
            ], axis=1) # x/init: [pos_x, pos_y, u_x, u_y, sdf, normal_x, normal_y] (7 dims)
            
            target = np.concatenate([
                internal.point_data["U"][:, :2],
                internal.point_data["p"][:, None],
                internal.point_data["nut"][:, None],
            ], axis=-1) # y/target: [v_x, v_y, p, nut] (4 dims)
            
            return pos, init, target, surf_bool


    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data_list_names)


    # 在 AirfRANSDataset 类中:

    def __getitem__(self, idx: int) -> Data:
        """
        获取单个样本，并完成所有预处理
        (加载, 归一化, 子采样, 构建图)
        """
        # 1. 获取文件名
        sim_name = self.data_list_names[idx]
        
        try:
            # 2. 加载原始数据
            pos, x, y, surf = self._load_single_simulation(sim_name)
            
            # 3. 归一化
            if self.coef_norm:
                mean_in, std_in, mean_out, std_out = self.coef_norm
                x = (x - mean_in) / (std_in + 1e-8)
                y = (y - mean_out) / (std_out + 1e-8)
                
            # 4. 转换为 Tensor
            pos = torch.tensor(pos, dtype=torch.float)
            x = torch.tensor(x, dtype=torch.float)
            y = torch.tensor(y, dtype=torch.float)
            surf = torch.tensor(surf, dtype=torch.bool)
            
            # 5. 子采样 (Subsampling)
            # 仅在 'train' 或 'val' 模式下进行子采样
            if self.mode in ['train', 'val']:
                subsampling_count = self.config.data.subsampling
                if pos.size(0) > subsampling_count:
                    idx_sample = random.sample(range(pos.size(0)), subsampling_count)
                    idx_sample = torch.tensor(idx_sample)
                    pos_s = pos[idx_sample]
                    x_s = x[idx_sample]
                    y_s = y[idx_sample]
                    surf_s = surf[idx_sample]
                elif pos.size(0) == subsampling_count:
                    pos_s, x_s, y_s, surf_s = pos, x, y, surf
                else:
                    self.logger.warning(
                        f"Sim {sim_name}: Loaded {pos.size(0)} points, "
                        f"which is less than subsampling count {subsampling_count}. "
                        "Using all loaded points."
                    )
                    pos_s, x_s, y_s, surf_s = pos, x, y, surf
            else:
                # 如果是 'test' 模式，则不进行子采样
                pos_s, x_s, y_s, surf_s = pos, x, y, surf
            
            # 6. 构建图结构
            model_hparams = self.config.model_hparams
            if model_hparams.build_graph:
                edge_index = nng.radius_graph(
                    x=pos_s, 
                    r=model_hparams.r, 
                    loop=True,
                    max_num_neighbors=int(model_hparams.max_neighbors)
                )
            else:
                edge_index = None

            # 7. 返回 PyG Data 对象
            return Data(
                pos=pos_s, 
                x=x_s, 
                y=y_s, 
                surf=surf_s, 
                edge_index=edge_index
            )
            
        except Exception as e:
            self.logger.error(f"Error loading data for index {idx} (sim: {sim_name}): {e}", exc_info=True)
            return Data()
    


class AirfRANSDatapipe:
    """
    为 AirfRANS (CFD) 数据集创建 DataLoader
    """
    
    def __init__(self, params, distributed: bool):
        self.params = params
        self.distributed = distributed

        self.train_dataset = AirfRANSDataset(config=self.params, mode='train')
        
        self.coef_norm = self.train_dataset.coef_norm
        
        self.val_dataset = AirfRANSDataset(config=self.params, mode='val', coef_norm=self.coef_norm)
        self.test_dataset = AirfRANSDataset(config=self.params, mode='test', coef_norm=self.coef_norm)

    def train_dataloader(self):
        sampler = DistributedSampler(self.train_dataset, shuffle=True) if self.distributed else None
        
        data_loader = PyGDataLoader(
            self.train_dataset,
            batch_size=self.params.dataloader.batch_size,
            drop_last=True if self.distributed else False,
            num_workers=self.params.dataloader.num_workers,
            pin_memory=True,
            shuffle=(sampler is None), 
            sampler=sampler
        )
        return data_loader, sampler

    def val_dataloader(self):
        sampler = DistributedSampler(self.val_dataset, shuffle=False) if self.distributed else None
        
        data_loader = PyGDataLoader(
            self.val_dataset,
            batch_size=self.params.dataloader.batch_size,
            drop_last=True if self.distributed else False,
            num_workers=self.params.dataloader.num_workers,
            pin_memory=True,
            shuffle=False,
            sampler=sampler
        )
        return data_loader, sampler

    def test_dataloader(self):
        # 测试集不使用分布式采样
        sampler = None
        
        data_loader = PyGDataLoader(
            self.test_dataset,
            batch_size=self.params.dataloader.batch_size,
            drop_last=False,
            num_workers=self.params.dataloader.num_workers,
            pin_memory=True,
            shuffle=False,
            sampler=sampler
        )
        return data_loader