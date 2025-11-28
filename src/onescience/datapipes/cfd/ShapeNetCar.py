# onescience/datapipes/car_dataset.py

import logging
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
from tqdm import tqdm
import vtk
import torch
import itertools
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data.distributed import DistributedSampler
import torch_geometric.nn as nng
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from onescience.datapipes.core import BaseDataset
from onescience.distributed.manager import DistributedManager
# 导入您旧代码中的辅助函数 (见下一个文件)s

def load_unstructured_grid_data(file_name):  # 加载VTK非结构化网格文件
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(file_name)
    reader.Update()
    output = reader.GetOutput()
    return output

def unstructured_grid_data_to_poly_data(
    unstructured_grid_data,
):  # 将非结构化网格转为表面多边形数据
    filter = vtk.vtkDataSetSurfaceFilter()
    filter.SetInputData(unstructured_grid_data)
    filter.Update()
    poly_data = filter.GetOutput()
    return poly_data, filter
def get_speed_from_poly_data(poly_data):  # 更改为从poly_data获取速度
    # 获取速度分量
    velo = vtk_to_numpy(poly_data.GetPointData().GetVectors())
    if velo.size == 0:
        raise ValueError("速度向量数据不存在于poly_data的PointData中")
    # 计算速度模长
    speed = np.linalg.norm(velo, axis=1)
    return speed
def get_sdf(target, boundary):  # 计算符号距离函数(SDF)
    nbrs = NearestNeighbors(n_neighbors=1).fit(boundary)
    dists, indices = nbrs.kneighbors(target)
    neis = np.array([boundary[i[0]] for i in indices])
    dirs = (target - neis) / (dists + 1e-8)
    return dists.reshape(-1), dirs


def get_normal(unstructured_grid_data):  # 计算网格点法线
    poly_data, surface_filter = unstructured_grid_data_to_poly_data(
        unstructured_grid_data
    )
    # visualize_poly_data(poly_data, surface_filter)
    # poly_data.GetPointData().SetScalars(None)
    normal_filter = vtk.vtkPolyDataNormals()
    normal_filter.SetInputData(poly_data)
    normal_filter.SetAutoOrientNormals(1)
    normal_filter.SetConsistency(1)
    # normal_filter.SetSplitting(0)
    normal_filter.SetComputeCellNormals(1)
    normal_filter.SetComputePointNormals(0)
    normal_filter.Update()
    """
    normal_filter.SetComputeCellNormals(0)
    normal_filter.SetComputePointNormals(1)
    normal_filter.Update()
    #visualize_poly_data(poly_data, surface_filter, normal_filter)
    poly_data.GetPointData().SetNormals(normal_filter.GetOutput().GetPointData().GetNormals())
    p2c = vtk.vtkPointDataToCellData()
    p2c.ProcessAllArraysOn()
    p2c.SetInputData(poly_data)
    p2c.Update()
    unstructured_grid_data.GetCellData().SetNormals(p2c.GetOutput().GetCellData().GetNormals())
    #visualize_poly_data(poly_data, surface_filter, p2c)
    """

    unstructured_grid_data.GetCellData().SetNormals(
        normal_filter.GetOutput().GetCellData().GetNormals()
    )
    c2p = vtk.vtkCellDataToPointData()
    # c2p.ProcessAllArraysOn()
    c2p.SetInputData(unstructured_grid_data)
    c2p.Update()
    unstructured_grid_data = c2p.GetOutput()
    # return unstructured_grid_data
    normal = vtk_to_numpy(c2p.GetOutput().GetPointData().GetNormals()).astype(np.double)
    # print(np.max(np.max(np.abs(normal), axis=1)), np.min(np.max(np.abs(normal), axis=1)))
    normal /= np.max(np.abs(normal), axis=1, keepdims=True) + 1e-8
    normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-8
    if np.isnan(normal).sum() > 0:
        print(np.isnan(normal).sum())
        print("recalculate")
        return get_normal(unstructured_grid_data)  # re-calculate
    # print(normal)
    return normal

def get_edges(unstructured_grid_data, points, cell_size=4):  # 提取网格单元边信息
    edge_indeces = set()
    cells = vtk_to_numpy(unstructured_grid_data.GetCells().GetData()).reshape(
        -1, cell_size + 1
    )
    for i in range(len(cells)):
        for j, k in itertools.product(range(1, cell_size + 1), repeat=2):
            edge_indeces.add((cells[i][j], cells[i][k]))
            edge_indeces.add((cells[i][k], cells[i][j]))
    edges = [[], []]
    for u, v in edge_indeces:
        edges[0].append(tuple(points[u]))
        edges[1].append(tuple(points[v]))
    return edges


def get_edge_index(pos, edges_press, edges_velo):  # 合并压力/速度场边信息
    indices = {tuple(pos[i]): i for i in range(len(pos))}
    edges = set()
    for i in range(len(edges_press[0])):
        edges.add((indices[edges_press[0][i]], indices[edges_press[1][i]]))
    for i in range(len(edges_velo[0])):
        edges.add((indices[edges_velo[0][i]], indices[edges_velo[1][i]]))
    edge_index = np.array(list(edges)).T
    return edge_index


def get_induced_graph(data, idx, num_hops):  # 提取子图
    subset, sub_edge_index, _, _ = k_hop_subgraph(
        node_idx=idx, num_hops=num_hops, edge_index=data.edge_index, relabel_nodes=True
    )
    return Data(x=data.x[subset], y=data.y[idx], edge_index=sub_edge_index)


def pc_normalize(pc):  # 点云归一化
    centroid = torch.mean(pc, axis=0)
    pc = pc - centroid
    m = torch.max(torch.sqrt(torch.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def get_shape(
    data, max_n_point=8192, normalize=True, use_height=False
):  # 提取表面形状特征
    surf_indices = torch.where(data.surf)[0].tolist()

    if len(surf_indices) > max_n_point:
        surf_indices = np.array(random.sample(range(len(surf_indices)), max_n_point))

    shape_pc = data.pos[surf_indices].clone()

    if normalize:
        shape_pc = pc_normalize(shape_pc)

    if use_height:
        gravity_dim = 1
        height_array = (
            shape_pc[:, gravity_dim : gravity_dim + 1]
            - shape_pc[:, gravity_dim : gravity_dim + 1].min()
        )
        shape_pc = torch.cat((shape_pc, height_array), axis=1)

    return shape_pc


def create_edge_index_radius(data, r, max_neighbors=32):  # 基于半径构建邻域图
    data.edge_index = nng.radius_graph(
        x=data.pos, r=r, loop=True, max_num_neighbors=max_neighbors
    )
    # print(f'r = {r}, #edges = {data.edge_index.size(1)}')
    return data


def get_samples(root):
    folds = [f'param{i}' for i in range(9)]
    samples = []
    for fold in folds:
        fold_samples = []
        files = os.listdir(os.path.join(root, fold))
        for file in files:
            path = os.path.join(root, os.path.join(fold, file))
            if os.path.isdir(path):
                fold_samples.append(os.path.join(fold, file))
        samples.append(fold_samples)
    return samples  # 100 + 99 + 97 + 100 + 100 + 96 + 100 + 98 + 99 = 889 samples    

class ShapeNetCarDataset(BaseDataset):
    """
    ShapeNetCar (CFD 汽车) 数据集
    
    继承自 BaseDataset，用于处理 PyTorch Geometric 的 Data 对象。
    """
    
    # 1. 覆盖元数据
    DOMAIN = "cfd"
    TASK = "regression"
    DATA_FORMATS = ["vtk", "npy"] # 支持原始vtk和预处理npy

    def __init__(self, config: Union[Dict[str, Any]], mode: str = 'train', coef_norm: Optional[Tuple] = None):
        """
        初始化 ShapeNetCar 数据集
        
        Parameters
        ----------
        config : Dict[str, Any]
            数据集配置 (来自 YAML 文件的 datapipe 部分)
        mode : str, optional
            'train', 'val', 或 'test'
        coef_norm : tuple, optional
            (mean_in, std_in, mean_out, std_out)
        """
        self.mode = mode
        self._provided_coef_norm = coef_norm
        self.data_list_names = []
        self.coef_norm = None
        self.dist = DistributedManager()
        
        super().__init__(config)
        if self.logger.hasHandlers():
            self.logger.handlers.clear() 

        # 处理图构建参数 
        self._init_graph_params()

        #初始化
        self._init_paths()
        self._filter_valid_samples()
        self._load_metadata() # 处理归一化
        
        if self.dist.rank == 0:
            self.logger.info(f"[{self.mode}] ShapeNetCar dataset initialized.")
            self.logger.info(f"[{self.mode}] Found {len(self.data_list_names)} simulation files.")

    def _init_paths(self):
        """
        加载样本列表并根据 mode 和 fold_id 拆分数据集
        """
        super()._init_paths() # self.data_path 已经设置
        
        # 从 config 中获取要使用的 fold_id
        fold_id = self.config.data.splits.fold_id

        all_samples_by_fold = get_samples(self.data_path)
        
        if not (0 <= fold_id < len(all_samples_by_fold)):
            raise ValueError(f"Invalid fold_id: {fold_id}. Must be between 0 and {len(all_samples_by_fold) - 1}")

        if self.mode == 'train':
            trainlst = []
            for i in range(len(all_samples_by_fold)):
                if i == fold_id:
                    continue
                trainlst += all_samples_by_fold[i]
            self.data_list_names = trainlst
        elif self.mode == 'val':
            self.data_list_names = all_samples_by_fold[fold_id]
        elif self.mode == 'test':
            # 假设测试集与验证集相同 (如果需要单独的测试集，请修改此逻辑)
            self.logger.warning(f"[{self.mode}] No dedicated test split found. Using validation fold {fold_id} as test set.")
            self.data_list_names = all_samples_by_fold[fold_id]
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def _init_graph_params(self):
        """
        1. 从 self.config.model_hparams (由训练脚本注入) 读取图构建参数。
        2. 将参数存储为实例属性 (self.use_cfd_mesh 等)。
        3. 在初始化时打印一次日志，说明将使用哪种图。
        """
        
        hparams_node = self.config.model_hparams
        
        if hasattr(hparams_node, 'to_dict'):
            hparams = hparams_node.to_dict()
        else:
            hparams = hparams_node # 假设它是一个类字典对象

        # 读取参数并存为实例属性
        self.use_cfd_mesh = hparams.get('cfd_mesh', False) # 默认 False
        self.radius_r = hparams.get('r', 0.2)
        self.radius_max_neighbors = int(hparams.get('max_neighbors', 32))

        # 只在 rank 0 打印一次日志
        if self.dist.rank == 0:
            if not self.use_cfd_mesh:
                self.logger.info(
                    f"[{self.mode}] Config 'cfd_mesh: False'. Datapipe将按需构建半径图 "
                    f"(r={self.radius_r}, max_neighbors={self.radius_max_neighbors})"
                )
            else:
                self.logger.info(f"[{self.mode}] Config 'cfd_mesh: True'. Datapipe将使用预加载的CFD网格图。")

    def _filter_valid_samples(self):
        """
        检查 self.data_list_names 中的每个样本，
        只保留那些所需文件（预处理的或原始VTK）真实存在的样本。
        """
        if self.dist.rank == 0:
            self.logger.info(f"[{self.mode}] Verifying sample file existence... (Initial count: {len(self.data_list_names)})")

        use_preprocessed = bool(self.config.source.get('preprocessed', False))
        preprocessed_dir = self.config.source.get('preprocessed_save_dir', None)
        root = self.data_path
        
        valid_samples = []
        
        pbar = self.data_list_names
        if self.dist.rank == 0:
            pbar = tqdm(self.data_list_names, desc=f"[{self.mode}] Verifying samples")
            
        for s in pbar:
            sample_is_valid = False
            
            # --- (FIX 1: 检查预处理文件) ---
            if use_preprocessed and preprocessed_dir:
                save_path = Path(preprocessed_dir) / s
                if (save_path / "x.npy").exists():
                    sample_is_valid = True
            
            if not sample_is_valid:
                file_name_press = root / s / "quadpress_smpl.vtk"
                file_name_velo = root / s / "hexvelo_smpl.vtk"
                if file_name_press.exists() and file_name_velo.exists():
                    sample_is_valid = True

            if sample_is_valid:
                valid_samples.append(s)
        
        if len(valid_samples) < len(self.data_list_names) and self.dist.rank == 0:
            self.logger.warning(f"[{self.mode}] Skipped {len(self.data_list_names) - len(valid_samples)} samples due to missing files.")

        self.data_list_names = valid_samples

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
            # 保存计算得到的统计数据
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
        """
        if self.mode != 'train':
            raise RuntimeError("Normalization calculation should only be done on the training set.")
        
        # --- (FIX: 临时禁用预处理保存) ---
        # 我们不希望在计算统计数据时写入数百个文件
        # 备份原始设置
        original_preprocessed_setting = self.config.source.get('preprocessed', False)
        # 临时关闭 "use_preprocessed" 会强制 _load_single_simulation 走 VTK 路径
        # 并且不会触发保存逻辑
        self.config.source['preprocessed'] = False 
        
        mean_in = None
        mean_out = None
        std_in = None
        std_out = None
        old_length = 0

        # 第一次遍历：计算均值
        self.logger.info("Calculating mean (stats pass 1/2)...")
        pbar = tqdm(self.data_list_names, desc="Norm (pass 1/2)")
        for s in pbar:
            # (现在 _load_single_simulation 会从 VTK 加载, 且不会保存)
            loaded_data = self._load_single_simulation(s)
            if loaded_data is None:
                continue 
            _, init, target, _, _ = loaded_data
            
            # ... (均值计算) ...
            if mean_in is None:
                mean_in = init.mean(axis=0, dtype=np.double)
                mean_out = target.mean(axis=0, dtype=np.double)
                old_length = init.shape[0]
            else:
                new_length = old_length + init.shape[0]
                mean_in += (init.sum(axis=0, dtype=np.double) - init.shape[0] * mean_in) / new_length
                mean_out += (target.sum(axis=0, dtype=np.double) - init.shape[0] * mean_out) / new_length
                old_length = new_length

        mean_in = mean_in.astype(np.single)
        mean_out = mean_out.astype(np.single)

        # 第二次遍历：计算标准差
        self.logger.info("Calculating std dev (stats pass 2/2)...")
        old_length = 0 # 重置
        pbar = tqdm(self.data_list_names, desc="Norm (pass 2/2)")
        for s in pbar:
            # (再次从 VTK 加载, 不保存)
            loaded_data = self._load_single_simulation(s)
            if loaded_data is None:
                continue 
            _, init, target, _, _ = loaded_data
            
            # ... (标准差计算) ...
            if std_in is None:
                old_length = init.shape[0] 
                std_in = ((init - mean_in) ** 2).sum(axis=0, dtype=np.double) / old_length
                std_out = ((target - mean_out) ** 2).sum(axis=0, dtype=np.double) / old_length
            else:
                new_length = old_length + init.shape[0]
                std_in += (((init - mean_in) ** 2).sum(axis=0, dtype=np.double) - init.shape[0] * std_in) / new_length
                std_out += (((target - mean_out) ** 2).sum(axis=0, dtype=np.double) - init.shape[0] * std_out) / new_length
                old_length = new_length
        
        # --- (FIX: 恢复原始设置) ---
        self.config.source['preprocessed'] = original_preprocessed_setting

        std_in = np.sqrt(std_in).astype(np.single)
        std_out = np.sqrt(std_out).astype(np.single)

        return (mean_in, std_in, mean_out, std_out)
        

    def _load_single_simulation(self, s: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        加载单个模拟，采用 "先检查NPY，后回退VTK" 逻辑
        """
        use_preprocessed = bool(self.config.source.get('preprocessed', False))
        preprocessed_dir = self.config.source.get('preprocessed_save_dir', None)
        root = self.data_path

        preprocessed_file_path = None
        save_path = None

        if use_preprocessed and preprocessed_dir:
            save_path = Path(preprocessed_dir) / s
            preprocessed_file_path = save_path / "x.npy" # 检查一个关键文件

        # 模式1: 尝试加载预处理数据
        if preprocessed_file_path and preprocessed_file_path.exists():
            try:
                init = np.load(save_path / "x.npy")
                target = np.load(save_path / "y.npy")
                pos = np.load(save_path / "pos.npy")
                surf = np.load(save_path / "surf.npy")
                edge_index = np.load(save_path / "edge_index.npy")
                return pos, init, target, surf, edge_index
            except Exception as e:
                self.logger.warning(f"加载预处理文件 {s} 失败: {e}。将从 VTK 重新处理。")
                # (继续执行到模式2)

        # 模式2: (如果预处理文件不存在 或 加载失败 或 use_preprocessed=False)
        # --- 从 VTK 处理 ---
        file_name_press = root / s / "quadpress_smpl.vtk"
        file_name_velo = root / s / "hexvelo_smpl.vtk"

        # (这个检查是安全的，因为 _filter_valid_samples 已经验证过)
        if not file_name_press.exists() or not file_name_velo.exists():
            self.logger.error(f"严重错误: VTK 文件 {s} 在 _filter_valid_samples 后消失了。")
            return None # __getitem__ 会捕获这个

        unstructured_grid_data_press = load_unstructured_grid_data(str(file_name_press))
        unstructured_grid_data_velo = load_unstructured_grid_data(str(file_name_velo))

        # ... (你所有的 VTK 处理逻辑) ...
        velo = vtk_to_numpy(unstructured_grid_data_velo.GetPointData().GetVectors())
        press = vtk_to_numpy(unstructured_grid_data_press.GetPointData().GetScalars())
        points_velo = vtk_to_numpy(unstructured_grid_data_velo.GetPoints().GetData())
        points_press = vtk_to_numpy(unstructured_grid_data_press.GetPoints().GetData())
        
        edges_press = get_edges(unstructured_grid_data_press, points_press, cell_size=4)
        edges_velo = get_edges(unstructured_grid_data_velo, points_velo, cell_size=8)
        
        sdf_velo, _ = get_sdf(points_velo, points_press) 
        sdf_press = np.zeros(points_press.shape[0])
        normal_press = get_normal(unstructured_grid_data_press)
        
        sdf_velo, normal_velo = get_sdf(points_velo, points_press)

        surface = {tuple(p) for p in points_press}
        exterior_indices = [i for i, p in enumerate(points_velo) if tuple(p) not in surface]
        velo_dict = {tuple(p): velo[i] for i, p in enumerate(points_velo)}

        pos_ext = points_velo[exterior_indices]
        pos_surf = points_press
        sdf_ext = sdf_velo[exterior_indices]
        sdf_surf = sdf_press
        normal_ext = normal_velo[exterior_indices]
        normal_surf = normal_press
        velo_ext = velo[exterior_indices]
        velo_surf = np.array([
            velo_dict[tuple(p)] if tuple(p) in velo_dict else np.zeros(3)
            for p in pos_surf
        ])
        press_ext = np.zeros([len(exterior_indices), 1])
        press_surf = press

        init_ext = np.c_[pos_ext, sdf_ext, normal_ext]
        init_surf = np.c_[pos_surf, sdf_surf, normal_surf]
        target_ext = np.c_[velo_ext, press_ext]
        target_surf = np.c_[velo_surf, press_surf]

        surf = np.concatenate([np.zeros(len(pos_ext)), np.ones(len(pos_surf))])
        pos = np.concatenate([pos_ext, pos_surf])
        init = np.concatenate([init_ext, init_surf])
        target = np.concatenate([target_ext, target_surf])
        edge_index = get_edge_index(pos, edges_press, edges_velo)
        # ... (VTK 处理结束) ...

        # (可选) 保存预处理结果
        # 仅当 use_preprocessed=True 且 preprocessed_dir 被设置时才保存
        if use_preprocessed and preprocessed_dir:
            if save_path is None: # 以防万一
                save_path = Path(preprocessed_dir) / s
            save_path.mkdir(parents=True, exist_ok=True)
            np.save(save_path / "x.npy", init)
            np.save(save_path / "y.npy", target)
            np.save(save_path / "pos.npy", pos)
            np.save(save_path / "surf.npy", surf)
            np.save(save_path / "edge_index.npy", edge_index)

        return pos, init, target, surf, edge_index

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data_list_names)

    def __getitem__(self, idx: int) -> Data:
        """
        获取单个样本，并完成所有预处理
        """
        sim_name = self.data_list_names[idx]

        try:
            # 加载
            pos, x, y, surf, edge_index = self._load_single_simulation(sim_name)

            # 归一化
            if self.coef_norm:
                mean_in, std_in, mean_out, std_out = self.coef_norm
                if x.shape[1] != mean_in.shape[0]:
                    self.logger.warning(f"Shape mismatch: x={x.shape[1]} != mean_in={mean_in.shape[0]}. Check normalization logic.")
                x = (x - mean_in) / (std_in + 1e-8)
                y = (y - mean_out) / (std_out + 1e-8)

            # 转换为 Tensor
            pos = torch.tensor(pos, dtype=torch.float32)
            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)
            surf = torch.tensor(surf, dtype=torch.bool)
            edge_index = torch.tensor(edge_index, dtype=torch.long) # CFD 网格图

            
            if not self.use_cfd_mesh:
                # 重建为半径图
                edge_index = nng.radius_graph(
                    x=pos,
                    r=self.radius_r,
                    loop=True,
                    max_num_neighbors=self.radius_max_neighbors
                )          

            # 返回 PyG Data 对象
            return Data(
                pos=pos,
                x=x,
                y=y,
                surf=surf,
                edge_index=edge_index
            )

        except Exception as e:
            self.logger.error(f"UNEXPECTED Error loading data for index {idx} (sim: {sim_name}): {e}", exc_info=True)
            return Data()
    

class ShapeNetCarDatapipe:
    """
    为 ShapeNetCar (CFD) 数据集创建 DataLoaders
    """
    
    def __init__(self, params, distributed: bool):
        self.params = params
        self.distributed = distributed
        
        # 1. 初始化训练数据集
        self.train_dataset = ShapeNetCarDataset(config=self.params, mode='train')
        
        # 2. 获取归一化系数
        self.coef_norm = self.train_dataset.coef_norm
        
        # 3. 初始化验证和测试数据集 (传入归一化系数)
        self.val_dataset = ShapeNetCarDataset(config=self.params, mode='val', coef_norm=self.coef_norm)
        # self.test_dataset = ShapeNetCarDataset(config=self.params, mode='test', coef_norm=self.coef_norm)

    def train_dataloader(self):
        sampler = DistributedSampler(self.train_dataset, shuffle=True) if self.distributed else None
        
        data_loader = PyGDataLoader(
            self.train_dataset,
            batch_size=self.params.dataloader.batch_size,
            drop_last=True if self.distributed else False,
            num_workers=self.params.dataloader.num_workers,
            pin_memory=True,
            shuffle=(sampler is None), # 如果没有 sampler 则 shuffle
            sampler=sampler
        )
        return data_loader, sampler

    def val_dataloader(self):
        sampler = DistributedSampler(self.val_dataset, shuffle=False) if self.distributed else None
        
        data_loader = PyGDataLoader(
            self.val_dataset,
            batch_size=self.params.dataloader.batch_size, # 验证时通常用bs=1
            drop_last=True if self.distributed else False,
            num_workers=self.params.dataloader.num_workers,
            pin_memory=True,
            shuffle=False,
            sampler=sampler
        )
        return data_loader, sampler

    # def test_dataloader(self):
    #     sampler = None
    #     data_loader = PyGDataLoader(
    #         self.test_dataset,
    #         batch_size=self.params.dataloader.batch_size,
    #         drop_last=False,
    #         ...
    #     )
    #     return data_loader