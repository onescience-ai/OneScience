import os
import torch
import numpy as np
import copy
import logging
from typing import Any, Dict, Optional, Union
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import DataLoader as GeoDataLoader
import matplotlib.tri as tri
from torchvision.transforms import GaussianBlur
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm  # [新增] 进度条
from torch.utils.data.distributed import DistributedSampler

from onescience.datapipes.core import BaseDataset
from onescience.distributed.manager import DistributedManager
from onescience.utils.beno.util import to_np_array
from onescience.utils.beno.utilities import GaussianNormalizer, MeshGenerator

class BENODataset(BaseDataset):
    """
    BENO Dataset for Elliptic PDE with Heterogeneous Graph support.
    Supports Caching to speed up loading.
    """
    DOMAIN = "cfd"
    DATA_FORMATS = ["npy"]

    def __init__(self, config: Union[Dict[str, Any]], mode: str = 'train'):
        self.mode = mode
        self.dist = DistributedManager()
        super().__init__(config)
        
        # 兼容 Hydra 的 DictConfig
        if hasattr(self.config, 'data'):
            self.data_cfg = self.config.data 
            self.source_cfg = self.config.source
        else:
            self.data_cfg = self.config['data']
            self.source_cfg = self.config['source']
        
        self.ntrain = self.data_cfg.ntrain
        self.ntest = self.data_cfg.ntest
        self.resolution = self.data_cfg.resolution
        self.ns = self.data_cfg.ns
        
        if self.dist.rank != 0:
            self.logger.setLevel(logging.WARNING)

        self._init_paths()
        
        # [优化] 检查是否有缓存文件
        cache_name = f"cached_{self.source_cfg.file_prefix}_{mode}_{self.ntrain if mode=='train' else self.ntest}.pt"
        self.cache_path = self.data_path / cache_name
        
        if self.cache_path.exists():
            self._load_from_cache()
        else:
            self._load_and_process_data()

    def _init_paths(self):
        super()._init_paths()
        prefix = self.source_cfg.file_prefix
        self.path_rhs = self.data_path / f"RHS_{prefix}_all.npy"
        self.path_sol = self.data_path / f"SOL_{prefix}_all.npy"
        self.path_bc = self.data_path / f"BC_{prefix}_all.npy"
        
        if not self.path_rhs.exists():
            raise FileNotFoundError(f"Data file not found: {self.path_rhs}")

    def _load_from_cache(self):
        if self.dist.rank == 0:
            self.logger.info(f"Loading cached data from {self.cache_path}...")
        
        cache = torch.load(self.cache_path)
        self.data_list = cache['data_list']
        self.u_normalizer = cache['u_normalizer']
        self.a_normalizer = cache['a_normalizer']
        # 其他需要的 normalizer 可以按需保存
        
        if self.dist.rank == 0:
            self.logger.info(f"Loaded {len(self.data_list)} samples from cache.")

    def _load_and_process_data(self):
        if self.dist.rank == 0:
            self.logger.info(f"Processing raw data from {self.data_path} (This may take a while)...")

        # Load Raw Data
        f_all = np.load(self.path_rhs)
        sol_all = np.load(self.path_sol)
        bc_all = np.load(self.path_bc)
        
        cells_state = f_all[:, :, 3]
        coord_all = f_all[:, :, 0:2]
        bc_euco = bc_all[:, :, 0:2]
        
        # BC Value Normalization
        bc_value = bc_all[:, :, 2].reshape(-1, 128, 1)
        bc_value = torch.tensor(bc_value)
        bc_value_train = bc_value[0:self.ntrain, :, :]
        bc_euco = torch.tensor(bc_euco)
        
        bcv_normalizer = GaussianNormalizer(bc_value_train)
        bc_value = bcv_normalizer.encode(bc_value)
        bc_euco = to_np_array(torch.cat([bc_euco, bc_value], dim=-1)) 
        
        # Input Field Processing
        gblur = GaussianBlur(kernel_size=5, sigma=5)
        all_a = f_all[:, :, 2]
        
        all_a_tensor = torch.tensor(all_a.reshape(all_a.shape[0], self.resolution, self.resolution))
        all_a_smooth = to_np_array(gblur(all_a_tensor).flatten(start_dim=1))
        
        all_a_reshape = all_a_smooth.reshape(-1, self.resolution, self.resolution)
        n = self.resolution ** 2
        
        all_a_gradx = np.concatenate([
            all_a_reshape[:, 1:2] - all_a_reshape[:, 0:1],
            (all_a_reshape[:, 2:] - all_a_reshape[:, :-2]) / 2,
            all_a_reshape[:, -1:] - all_a_reshape[:, -2:-1],
        ], 1).reshape(-1, n)
        
        all_a_grady = np.concatenate([
            all_a_reshape[:, :, 1:2] - all_a_reshape[:, :, 0:1],
            (all_a_reshape[:, :, 2:] - all_a_reshape[:, :, :-2]) / 2,
            all_a_reshape[:, :, -1:] - all_a_reshape[:, :, -2:-1],
        ], 2).reshape(-1, n)
        
        all_u = sol_all[:, :, 0]

        train_slice = slice(0, self.ntrain)
        test_slice = slice(self.ntrain, self.ntrain + self.ntest)
        
        train_a = torch.FloatTensor(all_a[train_slice])
        train_a_smooth = torch.FloatTensor(all_a_smooth[train_slice])
        train_a_gradx = torch.FloatTensor(all_a_gradx[train_slice])
        train_a_grady = torch.FloatTensor(all_a_grady[train_slice])
        train_u = torch.FloatTensor(all_u[train_slice])
        
        # Compute Normalizers (Train set statistics)
        indomain_a = np.array([])
        indomain_u = np.array([])
        
        # [优化] 使用 tqdm 显示进度
        if self.dist.rank == 0:
            self.logger.info("Computing statistics...")
            
        for j in range(self.ntrain):
            # [优化] 使用 numpy boolean masking 替代循环
            mask = cells_state[j] == 0
            indomain_u = np.append(indomain_u, sol_all[j][mask])
            indomain_a = np.append(indomain_a, f_all[j][mask][:, 2])
            
        indomain_u = torch.tensor(indomain_u)
        indomain_a = torch.tensor(indomain_a)
        
        self.a_normalizer = GaussianNormalizer(indomain_a)
        self.as_normalizer = GaussianNormalizer(train_a_smooth)
        self.agx_normalizer = GaussianNormalizer(train_a_gradx)
        self.agy_normalizer = GaussianNormalizer(train_a_grady)
        self.u_normalizer = GaussianNormalizer(indomain_u) 
        
        if self.mode == 'train':
            dataset_a = self.a_normalizer.encode(train_a)
            dataset_as = self.as_normalizer.encode(train_a_smooth)
            dataset_agx = self.agx_normalizer.encode(train_a_gradx)
            dataset_agy = self.agy_normalizer.encode(train_a_grady)
            dataset_u = self.u_normalizer.encode(train_u)
            bc_data = bc_euco[train_slice]
            sample_indices = range(self.ntrain)
            global_offset = 0
        else: 
            test_a = torch.FloatTensor(all_a[test_slice])
            test_as = torch.FloatTensor(all_a_smooth[test_slice])
            test_agx = torch.FloatTensor(all_a_gradx[test_slice])
            test_agy = torch.FloatTensor(all_a_grady[test_slice])
            test_u = torch.FloatTensor(all_u[test_slice])
            
            dataset_a = self.a_normalizer.encode(test_a)
            dataset_as = self.as_normalizer.encode(test_as)
            dataset_agx = self.agx_normalizer.encode(test_agx)
            dataset_agy = self.agy_normalizer.encode(test_agy)
            dataset_u = test_u 
            bc_data = bc_euco[test_slice]
            sample_indices = range(self.ntest)
            global_offset = self.ntrain 
        
        grid_input = f_all[-1, :, 0:2]
        meshgenerator = MeshGenerator([[0, 1], [0, 1]], [self.resolution, self.resolution], grid_input=grid_input)
        
        self.data_list = []
        
        # [优化] 使用 tqdm 进度条
        iterator = tqdm(sample_indices, desc=f"Building Graphs ({self.mode})", dynamic_ncols=True) if self.dist.rank == 0 else sample_indices
        
        for j in iterator:
            global_idx = j + global_offset
            
            # [优化] 1. 使用 Numpy 掩码快速筛选 indomain 索引，替代列表 remove
            # cells_state: 0=in-domain, !=0 out-domain
            mask_in_domain = cells_state[global_idx] == 0
            mesh_idx_temp = np.where(mask_in_domain)[0] 
            
            # 2. Distance to Boundary (Original Logic but slightly cleaner)
            dist2bd_x = np.array([0, 0])[np.newaxis, :]
            dist2bd_y = np.array([0, 0])[np.newaxis, :]
            curr_coord = coord_all[global_idx]
            curr_bc = bc_data[j] # [128, 3]
            
            # 这是一个瓶颈，但如果是复杂几何，这是必要的。如果是正方形区域，可以大大简化。
            # 为了保持通用性，我们保留原逻辑，但因为外层循环有了进度条，用户体验会好很多。
            # (如果确定是正方形区域，建议直接用坐标计算距离，速度快100倍)
            
            for p_idx in mesh_idx_temp:
                indomain_x = curr_coord[p_idx][0]
                indomain_y = curr_coord[p_idx][1]
                
                # Vectorized search for matching boundary points
                # 寻找 x 坐标相同的边界点 (Vertical distance)
                diff_x = np.abs(curr_bc[:, 0] - indomain_x)
                horizon_bd_y = np.where(diff_x < 1e-4)[0]
                
                if len(horizon_bd_y) >= 2:
                    dist_y_val = [
                        np.abs(curr_bc[horizon_bd_y[0], 1] - indomain_y),
                        np.abs(curr_bc[horizon_bd_y[1], 1] - indomain_y)
                    ]
                    dist2bd_y = np.vstack([dist2bd_y, np.array(dist_y_val)[np.newaxis, :]])
                else:
                    # Fallback for corner cases or errors in mesh
                    dist2bd_y = np.vstack([dist2bd_y, np.array([0,0])[np.newaxis, :]])

                # 寻找 y 坐标相同的边界点 (Horizontal distance)
                diff_y = np.abs(curr_bc[:, 1] - indomain_y)
                horizon_bd_x = np.where(diff_y < 1e-4)[0]
                
                if len(horizon_bd_x) >= 2:
                    dist_x_val = [
                        np.abs(curr_bc[horizon_bd_x[0], 0] - indomain_x),
                        np.abs(curr_bc[horizon_bd_x[1], 0] - indomain_x)
                    ]
                    dist2bd_x = np.vstack([dist2bd_x, np.array(dist_x_val)[np.newaxis, :]])
                else:
                    dist2bd_x = np.vstack([dist2bd_x, np.array([0,0])[np.newaxis, :]])
                
            dist2bd_y = torch.tensor(dist2bd_y[1:]).float()
            dist2bd_x = torch.tensor(dist2bd_x[1:]).float()
            
            # 3. Sample & Build Graph
            idx = meshgenerator.sample(mesh_idx_temp)
            grid = meshgenerator.get_grid()
            
            xx = to_np_array(grid[:, 0])
            yy = to_np_array(grid[:, 1])
            triang = tri.Triangulation(xx, yy)
            tri_edge = triang.edges
            
            edge_index = meshgenerator.ball_connectivity(ns=self.ns, tri_edge=tri_edge)
            edge_attr = meshgenerator.attributes(theta=dataset_a[j, :])
            
            # 4. Node Features
            node_feat = torch.cat([
                grid,
                dataset_a[j, idx].reshape(-1, 1),
                dataset_as[j, idx].reshape(-1, 1),
                dataset_agx[j, idx].reshape(-1, 1),
                dataset_agy[j, idx].reshape(-1, 1),
                dist2bd_x,
                dist2bd_y
            ], dim=1)
            
            node_feat_2 = torch.cat([grid, torch.zeros([grid.shape[0], 4]), dist2bd_x, dist2bd_y], dim=1)
            
            bd_coord = torch.tensor(curr_bc)
            bd_coord_zero = bd_coord.clone()
            bd_coord_zero[:, 2] = 0 
            
            cell_state_current = torch.FloatTensor(cells_state[global_idx])

            data = HeteroData()
            data["G1"].x = node_feat
            data["G1"].boundary = bd_coord_zero
            data["G1"].edge_features = edge_attr
            data["G1"].sample_idx = idx
            data["G1"].edge_index = edge_index
            data["G1"].cell_state = cell_state_current

            data["G2"].x = node_feat_2
            data["G2"].boundary = bd_coord
            data["G2"].edge_features = edge_attr
            data["G2"].sample_idx = idx
            data["G2"].edge_index = edge_index
            
            data["G1+2"].y = dataset_u[j, idx]
            
            self.data_list.append(data)
            
        # [优化] 保存缓存
        if self.dist.rank == 0:
            torch.save({
                'data_list': self.data_list,
                'u_normalizer': self.u_normalizer,
                'a_normalizer': self.a_normalizer
            }, self.cache_path)
            self.logger.info(f"Saved processed data to {self.cache_path}")

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int):
        return self.data_list[idx]


class BENODatapipe:
    def __init__(self, config: Dict[str, Any], distributed: bool = False):
        self.config = config
        self.distributed = distributed
        
        if isinstance(config, (DictConfig, dict)):
             datapipe_cfg = OmegaConf.to_container(config.datapipe, resolve=True) if isinstance(config.datapipe, DictConfig) else copy.deepcopy(config.datapipe)
        else:
             datapipe_cfg = config.datapipe

        self.train_dataset = BENODataset(datapipe_cfg, mode='train')
        self.test_dataset = BENODataset(datapipe_cfg, mode='test')
        
        self.u_normalizer = self.train_dataset.u_normalizer
        self.a_normalizer = self.train_dataset.a_normalizer

    def train_dataloader(self):
        loader_args = self.config.datapipe.dataloader
        sampler = DistributedSampler(self.train_dataset, shuffle=True) if self.distributed else None
        return GeoDataLoader(
            self.train_dataset,
            batch_size=loader_args.batch_size,
            sampler=sampler,
            num_workers=loader_args.num_workers,
            pin_memory=loader_args.pin_memory
        ), sampler

    def test_dataloader(self):
        loader_args = self.config.datapipe.dataloader
        sampler = DistributedSampler(self.test_dataset, shuffle=False) if self.distributed else None
        return GeoDataLoader(
            self.test_dataset,
            batch_size=loader_args.batch_size, 
            sampler=sampler,
            num_workers=loader_args.num_workers,
            pin_memory=loader_args.pin_memory
        ), sampler