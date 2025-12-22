import torch
import random
import json
import logging
import numpy as np
from pathlib import Path
from torch import Tensor
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import one_hot
from torch.utils.data.distributed import DistributedSampler
from typing import Union, List, Tuple, Optional, Dict, Any
import copy
from onescience.datapipes.core import BaseDataset
from onescience.distributed.manager import DistributedManager

def _faces_to_edges(faces: Tensor) -> Tensor:
    """将面(三角形)转换为边(索引对)"""
    edges = torch.cat([faces[:, :, :2], faces[:, :, 1:], faces[:, :, ::2]], dim=1)

    receivers, _ = torch.min(edges, dim=-1)
    senders, _ = torch.max(edges, dim=-1)

    packed_edges = torch.stack([senders, receivers], dim=-1).int()
    unique_edges = torch.unique(packed_edges, dim=1)
    unique_edges = torch.cat([unique_edges, torch.flip(unique_edges, dims=[-1])], dim=1)

    return unique_edges

class EagleDataset(BaseDataset):
    """
    Eagle 数据集 (CFD)
    用于处理时序图数据
    """
    
    DOMAIN = "cfd"
    TASK = "forecasting"
    DATA_FORMATS = ["npz"]

    pressure_mean: Tensor = torch.tensor([-0.8322, 4.6050]).view(-1, 2)
    pressure_std: Tensor = torch.tensor([7.4013, 9.7232]).view(-1, 2)
    velocity_mean: Tensor = torch.tensor([-0.0015, 0.2211]).view(-1, 2)
    velocity_std: Tensor = torch.tensor([1.7970, 2.0258]).view(-1, 2)

    def __init__(self, config: Union[Dict[str, Any]], mode: str = 'train'):
        self.mode = mode
        self.ep_paths = []
        
        super().__init__(config)
        
        try:
            self.w_len = self.config.data.window_length
            self.n_cluster = self.config.data.n_cluster
            self._type_as_onehot = self.config.data.type_as_onehot
            self._with_cells = self.config.data.with_cells
            self._use_normalized = self.config.data.normalized
            
            self.splits_path = Path(self.config.source.splits_dir)
            if self.n_cluster > 1:
                self.cluster_path = Path(self.config.source.cluster_dir)
            else:
                self.cluster_path = None

        except AttributeError as e:
            self.logger.error(f"Config 缺少必要的键: {e}")
            raise
            
        self.dist = DistributedManager()
        self._init_paths()
        self._load_metadata()
        
        if self.dist.rank == 0:
            self.logger.info(f"[{self.mode}] Eagle dataset initialized.")
            self.logger.info(f"[{self.mode}] Found {len(self.ep_paths)} simulation files.")
            self.logger.info(f"[{self.mode}] Window length: {self.w_len}, N_cluster: {self.n_cluster}")

    def _init_paths(self):
        """加载split文件并设置self.ep_paths"""
        super()._init_paths()
        
        if not self.splits_path.exists():
            raise FileNotFoundError(f"Splits path {self.splits_path} does not exist")
            
        split_file = self.splits_path / f"{self.mode}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file {split_file} does not exist")

        with open(split_file, "r") as f:
            self.ep_paths = [self.data_path / l.strip() for l in f.readlines()]

    def _load_metadata(self):
        """检查路径和断言"""
        if self.n_cluster > 1 and (self.cluster_path is None or not self.cluster_path.exists()):
            raise FileNotFoundError(f"Cluster path {self.cluster_path} does not exist, but n_cluster is {self.n_cluster}")
            
        assert self.n_cluster in [-1, 1, 10, 20, 30, 40], "Unknown number of clusters"
        assert self.w_len <= 990, "window length must be smaller than 990"
        
        self.logger.debug(f"[{self.mode}] Metadata checks passed.")

    def _load_from_npz(
        self, path: Path
    ) -> Tuple[NDArray, NDArray, NDArray, int, NDArray, NDArray]:
        """从npz文件加载模拟数据"""
        t = 0 if self.w_len == 990 else random.randint(0, 990 - self.w_len)
        t = 100 if self.mode != "train" and self.w_len != 990 else t
        data = np.load(path / "sim.npz", mmap_mode="r")

        mesh_pos = data["pointcloud"][t : t + self.w_len].copy()

        cells = np.load(path / "triangles.npy")
        cells = cells[t : t + self.w_len]

        Vx = data["VX"][t : t + self.w_len].copy()
        Vy = data["VY"][t : t + self.w_len].copy()

        Ps = data["PS"][t : t + self.w_len].copy()
        Pg = data["PG"][t : t + self.w_len].copy()

        velocity = np.stack([Vx, Vy], axis=-1)
        pressure = np.stack([Ps, Pg], axis=-1)
        node_type = data["mask"][t : t + self.w_len].copy()

        return mesh_pos, cells, node_type, t, velocity, pressure

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.ep_paths)

    def __getitem__(self, item) -> Dict[str, Any]:
        """获取单个样本"""
        try:
            mesh_pos, faces, node_type, t, velocity, pressure = self._load_from_npz(
                self.ep_paths[item]
            )
            faces = torch.from_numpy(faces).long()
            mesh_pos = torch.from_numpy(mesh_pos).float()
            velocity = torch.from_numpy(velocity).float()
            pressure = torch.from_numpy(pressure).float()
            edges = _faces_to_edges(faces)
            node_type = torch.from_numpy(node_type).long()

            if self._type_as_onehot:
                node_type = one_hot(node_type, num_classes=9).squeeze(-2)

            if self._use_normalized:
                velocity, pressure = self.normalize(velocity, pressure)

            output = {
                "mesh_pos": mesh_pos,
                "edges": edges,
                "velocity": velocity,
                "pressure": pressure,
                "node_type": node_type,
            }

            if self._with_cells:
                output["cells"] = faces

            if self.n_cluster != -1:
                cluster_file_path = self.cluster_path / self.ep_paths[item].relative_to(
                    self.data_path
                )

                if self.n_cluster == 1:
                    clusters = (
                        torch.arange(mesh_pos.shape[1] + 1)
                        .view(1, -1, 1)
                        .repeat(velocity.shape[0], 1, 1)
                    )
                else:
                    clusters_file = cluster_file_path / f"constrained_kmeans_{self.n_cluster}.npy"
                    if not clusters_file.exists():
                        raise FileNotFoundError(f"Cluster file not found: {clusters_file}")
                    clusters = np.load(
                        clusters_file,
                        mmap_mode="r",
                    )[t : t + self.w_len].copy()
                    clusters = torch.from_numpy(clusters).long()
                output["cluster"] = clusters
            
            return output
        
        except Exception as e:
            self.logger.error(f"Error loading data for index {item} (path: {self.ep_paths[item]}): {e}", exc_info=True)
            return {}

    def normalize(self, velocity: Tensor, pressure: Tensor) -> Tuple[Tensor, Tensor]:
        """归一化速度场和压力场"""
        p_mean, v_mean, p_std, v_std = self._stat_to_device(
            pressure.device, velocity.device
        )
        p_shape, v_shape = pressure.shape, velocity.shape
        pressure, velocity = pressure.reshape(-1, 2), velocity.reshape(-1, 2)
        pressure = (pressure - p_mean) / p_std
        velocity = (velocity - v_mean) / v_std
        return velocity.reshape(v_shape), pressure.reshape(p_shape)

    def denormalize(self, velocity: Tensor, pressure: Tensor) -> Tuple[Tensor, Tensor]:
        """反归一化速度场和压力场"""
        p_mean, v_mean, p_std, v_std = self._stat_to_device(
            pressure.device, velocity.device
        )
        p_shape, v_shape = pressure.shape, velocity.shape
        pressure, velocity = pressure.reshape(-1, 2), velocity.reshape(-1, 2)
        pressure = (pressure * p_std) + p_mean
        velocity = (velocity * v_std) + v_mean
        return velocity.reshape(v_shape), pressure.reshape(p_shape)

    def _stat_to_device(
        self, p_device: torch.device, v_device: torch.device
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """将数据集统计数据发送到给定设备"""
        return (
            self.pressure_mean.to(p_device),
            self.velocity_mean.to(v_device),
            self.pressure_std.to(p_device),
            self.velocity_std.to(v_device),
        )

def collate(x_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    将样本列表聚合为一个批次，并填充序列
    """
    x_list = [x for x in x_list if x]
    if not x_list:
        return {}

    N_max = max([x["mesh_pos"].shape[-2] for x in x_list])
    E_max = max([x["edges"].shape[-2] for x in x_list])
    C_max = max([x["cluster"].shape[-2] for x in x_list])
    
    has_cells = "cells" in x_list[0]
    if has_cells:
        N_cells_max = max([x["cells"].shape[-2] for x in x_list])

    for batch, x in enumerate(x_list):
        for key in ["mesh_pos", "velocity", "pressure"]:
            tensor = x[key]
            T, N, S = tensor.shape
            x[key] = torch.cat([tensor, torch.zeros(T, N_max - N + 1, S)], dim=1)

        tensor = x["node_type"]
        T, N, S = tensor.shape
        x["node_type"] = torch.cat([tensor, 2 * torch.ones(T, N_max - N + 1, S)], dim=1)

        x["cluster_mask"] = torch.ones_like(x["cluster"])
        x["cluster_mask"][x["cluster"] == -1] = 0
        x["cluster"][x["cluster"] == -1] = N_max

        if x["cluster"].shape[1] < C_max:
            c = x["cluster"].shape[1]
            x["cluster"] = torch.cat(
                [
                    x["cluster"],
                    N_max
                    * torch.ones(
                        x["cluster"].shape[0], C_max - c, x["cluster"].shape[-1]
                    ),
                ],
                dim=1,
            )
            x["cluster_mask"] = torch.cat(
                [
                    x["cluster_mask"],
                    torch.zeros(
                        x["cluster_mask"].shape[0], C_max - c, x["cluster"].shape[-1]
                    ),
                ],
                dim=1,
            )

        edges = x["edges"]
        T, E, S = edges.shape
        x["edges"] = torch.cat([edges, N_max * torch.ones(T, E_max - E + 1, S)], dim=1)
        
        if has_cells:
            tensor = x["cells"]
            T, N_c, S = tensor.shape
            padding = N_max * torch.ones(T, N_cells_max - N_c + 1, S, dtype=torch.long)
            x["cells"] = torch.cat([tensor.long(), padding], dim=1)
            
        x["mask"] = torch.cat([torch.ones(T, N), torch.zeros(T, N_max - N + 1)], dim=1)

    output = {key: torch.empty(1) for key in x_list[0].keys()}
    for key in output.keys():
        output[key] = torch.stack([x[key] for x in x_list], dim=0)
    return output

class EagleDatapipe:
    """
    为Eagle (CFD)数据集创建DataLoaders
    """
    
    def __init__(self, params: Any, distributed: bool):
        self.params = params
        self.distributed = distributed
        
        self.train_params = copy.deepcopy(params) 
        self.train_params.data.window_length = self.params.data.window_length_train
        self.train_dataset = EagleDataset(config=self.train_params, mode='train')
        
        self.val_params = copy.deepcopy(params)
        self.val_params.data.window_length = self.params.data.window_length_val
        self.val_dataset = EagleDataset(config=self.val_params, mode='valid')

        self.test_params = copy.deepcopy(params)
        self.test_params.data.window_length = self.params.data.window_length_test
        self.test_dataset = EagleDataset(config=self.test_params, mode='test')

    def train_dataloader(self) -> Tuple[DataLoader, Optional[DistributedSampler]]:
        sampler = DistributedSampler(self.train_dataset, shuffle=True) if self.distributed else None
        
        data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.params.dataloader.batch_size,
            drop_last=True if self.distributed else False,
            num_workers=self.params.dataloader.num_workers,
            pin_memory=True,
            shuffle=(sampler is None),
            sampler=sampler,
            collate_fn=collate
        )
        return data_loader, sampler

    def val_dataloader(self) -> Tuple[DataLoader, Optional[DistributedSampler]]:
        sampler = DistributedSampler(self.val_dataset, shuffle=False) if self.distributed else None
        
        data_loader = DataLoader(
            self.val_dataset,
            batch_size=self.params.dataloader.batch_size,
            drop_last=True if self.distributed else False,
            num_workers=self.params.dataloader.num_workers,
            pin_memory=True,
            shuffle=False,
            sampler=sampler,
            collate_fn=collate
        )
        return data_loader, sampler

    def test_dataloader(self, batch_size: Optional[int] = None) -> Tuple[DataLoader, Optional[DistributedSampler]]:
        sampler = DistributedSampler(self.test_dataset, shuffle=False) if self.distributed else None
        
        final_batch_size = batch_size if batch_size is not None else self.params.dataloader.batch_size
        
        data_loader = DataLoader(
            self.test_dataset,
            batch_size=final_batch_size, 
            drop_last=False,
            num_workers=self.params.dataloader.num_workers,
            pin_memory=True,
            shuffle=False,
            sampler=sampler,
            collate_fn=collate
        )
        return data_loader, sampler