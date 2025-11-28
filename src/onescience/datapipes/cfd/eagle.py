"""
Eagle Datapipe 和 Dataset
继承自 BaseDataset 并提供 Dataloader 工厂
"""

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
# 假设 BaseDataset 和 DistributedManager 在这些路径
from onescience.datapipes.core import BaseDataset
from onescience.distributed.manager import DistributedManager

# --- EagleDataset 的辅助函数 ---
# (这些函数保持不变，但我们会将它们作为 EagleDataset 的私有方法)

def _faces_to_edges(faces: Tensor) -> Tensor:
    """将面(三角形)转换-边(索引对)。(静态辅助函数)"""
    edges = torch.cat([faces[:, :, :2], faces[:, :, 1:], faces[:, :, ::2]], dim=1)

    receivers, _ = torch.min(edges, dim=-1)
    senders, _ = torch.max(edges, dim=-1)

    packed_edges = torch.stack([senders, receivers], dim=-1).int()
    unique_edges = torch.unique(packed_edges, dim=1)
    unique_edges = torch.cat([unique_edges, torch.flip(unique_edges, dims=[-1])], dim=1)

    return unique_edges

# --- 重构后的 EagleDataset ---

class EagleDataset(BaseDataset):
    """
    Eagle 数据集 (CFD)
    
    继承自 BaseDataset，用于处理时序图数据。
    """
    
    # 1. 覆盖元数据
    DOMAIN = "cfd"
    TASK = "forecasting"
    DATA_FORMATS = ["npz"]

    # (硬编码的统计数据，来自原始类)
    pressure_mean: Tensor = torch.tensor([-0.8322, 4.6050]).view(-1, 2)
    pressure_std: Tensor = torch.tensor([7.4013, 9.7232]).view(-1, 2)
    velocity_mean: Tensor = torch.tensor([-0.0015, 0.2211]).view(-1, 2)
    velocity_std: Tensor = torch.tensor([1.7970, 2.0258]).view(-1, 2)

    def __init__(self, config: Union[Dict[str, Any]], mode: str = 'train'):
        """
        初始化 Eagle 数据集
        
        Parameters
        ----------
        config : Dict[str, Any]
            数据集配置 (来自 YAML 文件的 datapipe 部分)
        mode : str, optional
            'train', 'val', 或 'test'
        """
        self.mode = mode
        self.ep_paths = []
        
        # 2. 调用父类 __init__
        # 注意：我们传递 config 对象，而不是 config.data
        super().__init__(config) 
        
        # 3. 从 config 加载参数
        # (config 是 YParams 节点, 可以像属性一样访问)
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
                self.cluster_path = None # 确保 cluster_path 存在

        except AttributeError as e:
            self.logger.error(f"Config 缺少必要的键: {e}")
            raise
            
        # 4. 初始化
        self.dist = DistributedManager()
        self._init_paths()
        self._load_metadata()
        
        if self.dist.rank == 0:
            self.logger.info(f"[{self.mode}] Eagle dataset initialized.")
            self.logger.info(f"[{self.mode}] Found {len(self.ep_paths)} simulation files.")
            self.logger.info(f"[{self.mode}] Window length: {self.w_len}, N_cluster: {self.n_cluster}")

    def _init_paths(self):
        """
        加载 split 文件并设置 self.ep_paths
        """
        # super()._init_paths() 会检查 self.config.source.data_dir
        # 我们在这里使用它来设置 self.data_path
        super()._init_paths() 
        
        if not self.splits_path.exists():
            raise FileNotFoundError(f"Splits path {self.splits_path} does not exist")
            
        split_file = self.splits_path / f"{self.mode}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file {split_file} does not exist")

        with open(split_file, "r") as f:
            self.ep_paths = [self.data_path / l.strip() for l in f.readlines()]

    def _load_metadata(self):
        """
        检查路径和断言。
        AirfRANS 在这里计算归一化，但 Eagle 是硬编码的，所以我们只需检查。
        """
        if self.n_cluster > 1 and (self.cluster_path is None or not self.cluster_path.exists()):
            raise FileNotFoundError(f"Cluster path {self.cluster_path} does not exist, but n_cluster is {self.n_cluster}")
            
        assert self.n_cluster in [-1, 1, 10, 20, 30, 40], "Unknown number of clusters"
        assert self.w_len <= 990, "window length must be smaller than 990"
        
        # 元数据已在类级别硬编码
        self.logger.debug(f"[{self.mode}] Metadata checks passed.")


    def _load_from_npz(
        self, path: Path
    ) -> Tuple[NDArray, NDArray, NDArray, int, NDArray, NDArray]:
        """
        从 npz 文件加载模拟数据 (来自原始脚本)
        使用 self.w_len 和 self.mode
        """
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
        """获取单个样本 (来自原始脚本)"""
        try:
            mesh_pos, faces, node_type, t, velocity, pressure = self._load_from_npz(
                self.ep_paths[item]
            )
            faces = torch.from_numpy(faces).long()
            mesh_pos = torch.from_numpy(mesh_pos).float()
            velocity = torch.from_numpy(velocity).float()
            pressure = torch.from_numpy(pressure).float()
            edges = _faces_to_edges(faces)  # 使用静态辅助函数
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
            # 返回一个空字典或根据需要处理
            return {}


    # --- 归一化方法 (来自原始脚本) ---
    
    def normalize(self, velocity: Tensor, pressure: Tensor) -> Tuple[Tensor, Tensor]:
        """使用数据集统计数据归一化速度场和压力场。"""
        p_mean, v_mean, p_std, v_std = self._stat_to_device(
            pressure.device, velocity.device
        )
        p_shape, v_shape = pressure.shape, velocity.shape
        pressure, velocity = pressure.reshape(-1, 2), velocity.reshape(-1, 2)
        pressure = (pressure - p_mean) / p_std
        velocity = (velocity - v_mean) / v_std
        return velocity.reshape(v_shape), pressure.reshape(p_shape)

    def denormalize(self, velocity: Tensor, pressure: Tensor) -> Tuple[Tensor, Tensor]:
        """使用数据集统计数据反归一化速度场和压力场。"""
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
        """将数据集统计数据发送到给定的设备。"""
        return (
            self.pressure_mean.to(p_device),
            self.velocity_mean.to(v_device),
            self.pressure_std.to(p_device),
            self.velocity_std.to(v_device),
        )

def collate(x_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    将样本列表聚合为一个批次，并填充序列。
    (此函数来自原始脚本，保持不变)
    """
    # 过滤掉加载失败的样本 (空字典)
    x_list = [x for x in x_list if x]
    if not x_list:
        return {}

    # Find the maximum number of nodes and edges in the batch
    N_max = max([x["mesh_pos"].shape[-2] for x in x_list])
    E_max = max([x["edges"].shape[-2] for x in x_list])
    C_max = max([x["cluster"].shape[-2] for x in x_list])
    
    # --- 新增：检查 'cells' 并找到 N_cells_max ---
    has_cells = "cells" in x_list[0]
    if has_cells:
        N_cells_max = max([x["cells"].shape[-2] for x in x_list])
    # --- 结束新增 ---

    for batch, x in enumerate(x_list):
        # This step add fantom nodes to reach N_max + 1 nodes
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
        
        # --- 新增：为 'cells' 添加填充逻辑 ---
        if has_cells:
            tensor = x["cells"]
            T, N_c, S = tensor.shape  # T=Time, N_c=Num Cells, S=3
            
            # 使用鬼节点索引 (N_max) 进行填充
            # 确保填充张量的类型为 torch.long
            padding = N_max * torch.ones(T, N_cells_max - N_c + 1, S, dtype=torch.long)
            
            # 确保原始张量也是 long (如果它还不是)
            x["cells"] = torch.cat([tensor.long(), padding], dim=1)
        # --- 结束新增 ---
            
        x["mask"] = torch.cat([torch.ones(T, N), torch.zeros(T, N_max - N + 1)], dim=1)

    output = {key: torch.empty(1) for key in x_list[0].keys()}
    for key in output.keys():
        output[key] = torch.stack([x[key] for x in x_list], dim=0)
    return output

class EagleDatapipe:
    """
    为 Eagle (CFD) 数据集创建 DataLoaders
    """
    
    def __init__(self, params: Any, distributed: bool):
        """
        params: 来自 YAML 文件的 "datapipe" YParams 节点
        distributed: 是否启用 DDP
        """
        self.params = params
        self.distributed = distributed
        
        # 1. 初始化训练数据集
        # (使用 copy.deepcopy 
        #  YParams 对象)
        
        # 修正: 'YParams' object has no attribute 'copy'
        self.train_params = copy.deepcopy(params) 
        self.train_params.data.window_length = self.params.data.window_length_train
        self.train_dataset = EagleDataset(config=self.train_params, mode='train')
        
        # 2. 初始化验证数据集
        # 修正:
        self.val_params = copy.deepcopy(params)
        self.val_params.data.window_length = self.params.data.window_length_val
        self.val_dataset = EagleDataset(config=self.val_params, mode='valid')

        # 3. 初始化测试数据集
        # 修正:
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
            shuffle=(sampler is None), # 如果没有 sampler 则 shuffle
            sampler=sampler,
            collate_fn=collate # 使用自定义的 collate
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
            collate_fn=collate # 使用自定义的 collate
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
            collate_fn=collate # 使用自定义的 collate
        )
        return data_loader, sampler