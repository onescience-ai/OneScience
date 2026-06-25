import functools
import json
import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

# --- OneScience 核心 ---
from onescience.datapipes.core import BaseDataset 
from onescience.distributed.adapter import create_adapter
from onescience.distributed.megatron.core import mpu

# --- DGL 库特定导入 ---
try:
    import dgl
    from dgl.dataloading import GraphDataLoader
    from dgl import DGLGraph
except ImportError:
    raise ImportError(
        "此 DGL 版本的 Datapipe 需要 DGL 库。"
        "请访问: https://www.dgl.ai/pages/start.html"
    )

# --- TensorFlow (仅用于数据加载) ---
try:
    import tensorflow.compat.v1 as tf
    # 隐藏 GPU，避免 TF 占用 PyTorch 的显存
    tf.config.set_visible_devices([], "GPU") 
except ImportError:
    raise ImportError(
        "DeepMind_CylinderFlowDataset 需要 Tensorflow 库。"
        "请安装: pip install tensorflow"
    )


# --- 辅助函数 ---
def _save_json(stats: Dict[str, Any], path: Union[str, Path]):
    """
    将字典保存为 JSON 文件，自动处理 Torch Tensors。
    创建一个新字典来保存，避免修改原始字典。
    """
    stats_to_save = {}
    for key, value in stats.items():
        if isinstance(value, torch.Tensor):
            stats_to_save[key] = value.tolist()
        else:
            stats_to_save[key] = value

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fp:
        json.dump(stats_to_save, fp, indent=4) 
    logging.info(f"Saved stats to {path}")

def _load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """从 JSON 文件加载字典，并将列表转换回 Tensors。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Stats file not found: {path}")
        
    with open(path, "r") as fp:
        stats = json.load(fp)
    
    for key, value in stats.items():
        if isinstance(value, list):
            stats[key] = torch.tensor(value, dtype=torch.float32)
    logging.info(f"Loaded stats from {path}")
    return stats


class DeepMind_CylinderFlowDataset(BaseDataset):
    """
    DeepMind 圆柱绕流 (Vortex Shedding) 数据集 [DGL]
    
    继承自 BaseDataset，用于处理 DGL 图数据。
    封装了原始 VortexSheddingDataset 的 TFRecord 加载和预处理逻辑。
    """
    
    # 覆盖元数据
    DOMAIN = "cfd"
    TASK = "regression"
    DATA_FORMATS = ["tfrecord"]

    def __init__(
        self, 
        config: Dict[str, Any], 
        mode: str = 'train', 
        stats: Optional[Dict[str, torch.Tensor]] = None
    ):
        """
        初始化 DeepMind 圆柱绕流数据集
        
        Parameters
        ----------
        config : Dict[str, Any]
            数据集配置 (来自 YAML 文件的 datapipe 部分)
        mode : str, optional
            'train', 'val', 或 'test'
        stats : dict, optional
            包含 edge_stats 和 node_stats 的字典
        """
        self.mode = mode
        # 将 'val' 模式映射到 TFRecord 的 'valid' split
        self.split = 'valid' if mode == 'val' else mode
        
        self._provided_stats = stats
        self.stats = {} 
        
        # 从 config 中获取特定于 split 的参数
        if self.mode == 'train':
            self.num_samples = config.data.train_samples
            self.num_steps = config.data.train_steps
            self.noise_std = config.data.noise_std
        elif self.mode == 'val':
            self.num_samples = config.data.val_samples
            self.num_steps = config.data.val_steps
            self.noise_std = 0.0
        else: # 'test'
            self.num_samples = config.data.test_samples
            self.num_steps = config.data.test_steps
            self.noise_std = 0.0
            
        self.length = self.num_samples * (self.num_steps - 1)
        
        # 创建分布式适配器
        self.dist = create_adapter()
        
        super().__init__(config)
        
        if self.dist.is_rank0():
            self.logger.info(f"[{self.mode}] Initializing DeepMind CylinderFlow Dataset (DGL)...")
            self.logger.info(f"[{self.mode}] Environment: {self.dist.env_type}")
            self.logger.info(f"[{self.mode}] Mode='{self.mode}' (Split='{self.split}')")
            self.logger.info(f"[{self.mode}] Samples={self.num_samples}, Steps={self.num_steps}")

        self._init_paths()
        self._load_metadata() # 加载或计算归一化统计数据
        self._init_data()     # 加载和处理 TFRecord 数据
        
        if self.dist.is_rank0():
            self.logger.info(f"[{self.mode}] Dataset initialized. Total items: {self.length}")

    def _init_paths(self):
        """初始化数据路径并加载 meta.json"""
        super()._init_paths() 
        self.meta_path = self.data_path / "meta.json"
        
        if not self.meta_path.exists():
            raise FileNotFoundError(f"meta.json not found at: {self.data_path}")
            
        with open(self.meta_path, 'r') as fp:
            self.meta = json.load(fp)
        self.stats_dir = Path(self.config.source.stats_dir)
        self.stats_dir.mkdir(parents=True, exist_ok=True)
        self.edge_stats_path = self.stats_dir / "edge_stats.json"
        self.node_stats_path = self.stats_dir / "node_stats.json"

    def _load_metadata(self):
        """加载或计算归一化统计数据 (edge_stats, node_stats)"""
        if self._provided_stats:
            if self.dist.is_rank0():
                self.logger.debug(f"[{self.mode}] Using provided normalization stats.")
            self.stats = self._provided_stats
            return

        if self.mode == 'train':
            if self.edge_stats_path.exists() and self.node_stats_path.exists():
                if self.dist.is_rank0():
                    self.logger.info(f"[{self.mode}] Loading normalization stats from {self.stats_dir}")
                self.stats['edge_stats'] = _load_json(self.edge_stats_path)
                self.stats['node_stats'] = _load_json(self.node_stats_path)
            else:
                if self.dist.is_rank0():
                    self.logger.warning(f"[{self.mode}] Stats not found. Calculating stats on the fly...")
                pass 
        else:
            if not self.edge_stats_path.exists() or not self.node_stats_path.exists():
                raise FileNotFoundError(
                    f"[{self.mode}] Normalization stats not found in {self.stats_dir}. "
                    "Please run training mode first to generate stats."
                )
            if self.dist.is_rank0():
                self.logger.info(f"[{self.mode}] Loading normalization stats from {self.stats_dir}")
            self.stats['edge_stats'] = _load_json(self.edge_stats_path)
            self.stats['node_stats'] = _load_json(self.node_stats_path)


    def _init_data(self):
        """加载和处理 TFRecord 数据"""
        
        # --- 1. 加载图结构 (Edges) ---
        dataset_iterator_graphs = self._load_tf_data(self.data_path, self.split)
        self.graphs, self.cells, self.node_type = [], [], []
        noise_mask, self.rollout_mask = [], []
        self.mesh_pos = []
        
        if self.dist.is_rank0():
             self.logger.debug(f"[{self.mode}] Pass 1/2: Loading graph structure and edge features...")

        for i in range(self.num_samples):
            data_np = dataset_iterator_graphs.get_next()
            data_np = {key: arr[:self.num_steps].numpy() for key, arr in data_np.items()}
            
            src, dst = self._cell_to_adj(data_np["cells"][0])
            graph = self._create_graph(src, dst) 
            graph = self._add_edge_features(graph, data_np["mesh_pos"][0]) 
            self.graphs.append(graph)
            
            node_type_tensor = torch.tensor(data_np["node_type"][0], dtype=torch.uint8)
            self.node_type.append(self._one_hot_encode(node_type_tensor))
            noise_mask.append(torch.eq(node_type_tensor, torch.zeros_like(node_type_tensor)))

            if self.mode != "train":
                self.mesh_pos.append(torch.tensor(data_np["mesh_pos"][0]))
                self.cells.append(data_np["cells"][0])
                self.rollout_mask.append(self._get_rollout_mask(node_type_tensor))

        # --- 2. 边特征归一化 (计算或应用) ---
        if self.mode == "train" and 'edge_stats' not in self.stats:
            self.stats['edge_stats'] = self._get_edge_stats() 
            _save_json(self.stats['edge_stats'], self.edge_stats_path)
            if self.dist.is_rank0():
                self.logger.info(f"[{self.mode}] Saved edge stats to {self.edge_stats_path}")

        edge_mean = self.stats['edge_stats']["edge_mean"]
        edge_std = self.stats['edge_stats']["edge_std"]
        for i in range(self.num_samples):
            self.graphs[i].edata["x"] = self._normalize(
                self.graphs[i].edata["x"], edge_mean, edge_std
            )

        # --- 3. 加载节点特征 (Nodes) ---
        dataset_iterator_nodes = self._load_tf_data(self.data_path, self.split)
        self.node_features, self.node_targets = [], []
        
        if self.dist.is_rank0():
             self.logger.debug(f"[{self.mode}] Pass 2/2: Loading node features and targets...")

        for i in range(self.num_samples):
            data_np = dataset_iterator_nodes.get_next()
            data_np = {key: arr[:self.num_steps].numpy() for key, arr in data_np.items()}
            features, targets = {}, {}
            
            features["velocity"] = self._drop_last(data_np["velocity"])
            targets["velocity"] = self._push_forward_diff(data_np["velocity"])
            targets["pressure"] = self._push_forward(data_np["pressure"])

            if self.mode == "train":
                features["velocity"], targets["velocity"] = self._add_noise(
                    features["velocity"],
                    targets["velocity"],
                    self.noise_std,
                    noise_mask[i],
                )
            self.node_features.append(features)
            self.node_targets.append(targets)

        # --- 4. 节点特征归一化 (计算或应用) ---
        if self.mode == "train" and 'node_stats' not in self.stats:
            self.stats['node_stats'] = self._get_node_stats()
            _save_json(self.stats['node_stats'], self.node_stats_path)
            if self.dist.is_rank0():
                self.logger.info(f"[{self.mode}] Saved node stats to {self.node_stats_path}")

        node_stats = self.stats['node_stats']
        for i in range(self.num_samples):
            self.node_features[i]["velocity"] = self._normalize(
                self.node_features[i]["velocity"],
                node_stats["velocity_mean"],
                node_stats["velocity_std"],
            )
            self.node_targets[i]["velocity"] = self._normalize(
                self.node_targets[i]["velocity"],
                node_stats["velocity_diff_mean"],
                node_stats["velocity_diff_std"],
            )
            self.node_targets[i]["pressure"] = self._normalize(
                self.node_targets[i]["pressure"],
                node_stats["pressure_mean"],
                node_stats["pressure_std"],
            )

    def __len__(self) -> int:
        """返回数据集样本总数"""
        return self.length

    def __getitem__(self, idx: int) -> Union[DGLGraph, Tuple[DGLGraph, np.ndarray, torch.Tensor]]:
        """
        获取单个样本 (一个时间步的 DGL 图)
        """
        try:
            gidx = idx // (self.num_steps - 1)  # 图/模拟 索引
            tidx = idx % (self.num_steps - 1)  # 时间步 索引
            
            graph = self.graphs[gidx].clone()

            node_features = torch.cat(
                (self.node_features[gidx]["velocity"][tidx], self.node_type[gidx]), dim=-1
            )
            node_targets = torch.cat(
                (
                    self.node_targets[gidx]["velocity"][tidx],
                    self.node_targets[gidx]["pressure"][tidx],
                ),
                dim=-1,
            )
            
            graph.ndata["x"] = node_features
            graph.ndata["y"] = node_targets

            if self.mode == "train":
                # 训练模式：仅返回图
                return graph
            else:
                # 推理/验证模式：返回 (graph, cells, mask)
                graph.ndata["mesh_pos"] = self.mesh_pos[gidx]
                cells = self.cells[gidx]           # (numpy.ndarray)
                mask = self.rollout_mask[gidx]     # (torch.Tensor)
                return graph, cells, mask
                
        except Exception as e:
            self.logger.error(f"Error loading data for index {idx} (gidx: {gidx}, tidx: {tidx}): {e}", exc_info=True)
            # 发生异常时返回空对象以防止 crash
            if self.mode == "train":
                return dgl.graph(([], []))
            else:
                return dgl.graph(([], [])), np.array([]), torch.tensor([])

    # --- 统计数据计算 ---

    def _get_edge_stats(self) -> Dict[str, torch.Tensor]:
        """计算边特征的均值和标准差"""
        if self.dist.is_rank0():
            self.logger.info(f"[{self.mode}] Calculating edge stats...")
            
        stats = {
            "edge_mean": 0,
            "edge_meansqr": 0,
        }
        for i in range(self.num_samples):
            stats["edge_mean"] += (
                torch.mean(self.graphs[i].edata["x"], dim=0) / self.num_samples
            )
            stats["edge_meansqr"] += (
                torch.mean(torch.square(self.graphs[i].edata["x"]), dim=0)
                / self.num_samples
            )
        stats["edge_std"] = torch.sqrt(
            stats["edge_meansqr"] - torch.square(stats["edge_mean"])
        )
        stats.pop("edge_meansqr")
        return stats

    def _get_node_stats(self) -> Dict[str, torch.Tensor]:
        """计算节点特征的均值和标准差"""
        if self.dist.is_rank0():
            self.logger.info(f"[{self.mode}] Calculating node stats...")
        
        stats = {
            "velocity_mean": 0, "velocity_meansqr": 0,
            "velocity_diff_mean": 0, "velocity_diff_meansqr": 0,
            "pressure_mean": 0, "pressure_meansqr": 0,
        }
        for i in range(self.num_samples):
            stats["velocity_mean"] += (
                torch.mean(self.node_features[i]["velocity"], dim=(0, 1))
                / self.num_samples
            )
            stats["velocity_meansqr"] += (
                torch.mean(torch.square(self.node_features[i]["velocity"]), dim=(0, 1))
                / self.num_samples
            )
            stats["pressure_mean"] += (
                torch.mean(self.node_targets[i]["pressure"], dim=(0, 1))
                / self.num_samples
            )
            stats["pressure_meansqr"] += (
                torch.mean(torch.square(self.node_targets[i]["pressure"]), dim=(0, 1))
                / self.num_samples
            )
            stats["velocity_diff_mean"] += (
                torch.mean(self.node_targets[i]["velocity"], dim=(0, 1),)
                / self.num_samples
            )
            stats["velocity_diff_meansqr"] += (
                torch.mean(torch.square(self.node_targets[i]["velocity"]), dim=(0, 1),)
                / self.num_samples
            )
            
        stats["velocity_std"] = torch.sqrt(
            stats["velocity_meansqr"] - torch.square(stats["velocity_mean"])
        )
        stats["pressure_std"] = torch.sqrt(
            stats["pressure_meansqr"] - torch.square(stats["pressure_mean"])
        )
        stats["velocity_diff_std"] = torch.sqrt(
            stats["velocity_diff_meansqr"] - torch.square(stats["velocity_diff_mean"])
        )
        stats.pop("velocity_meansqr")
        stats.pop("pressure_meansqr")
        stats.pop("velocity_diff_meansqr")
        return stats


    # --- TFRecord 和图处理辅助函数 ---

    def _load_tf_data(self, path: Path, split: str) -> tf.data.Iterator:
        """加载 .tfrecord 数据集"""
        tf_path = str(path / f"{split}.tfrecord")
        if not os.path.exists(tf_path):
             raise FileNotFoundError(f"TFRecord file not found: {tf_path}")
             
        dataset = tf.data.TFRecordDataset(tf_path)
        dataset = dataset.map(
            functools.partial(self._parse_data, meta=self.meta), num_parallel_calls=8
        ).prefetch(tf.data.AUTOTUNE)
        return tf.data.make_one_shot_iterator(dataset)

    @staticmethod
    def _parse_data(p, meta):
        """解析 TFRecord 中的单个样本"""
        outvar = {}
        feature_dict = {k: tf.io.VarLenFeature(tf.string) for k in meta["field_names"]}
        features = tf.io.parse_single_example(p, feature_dict)
        for k, v in meta["features"].items():
            data = tf.reshape(
                tf.io.decode_raw(features[k].values, getattr(tf, v["dtype"])),
                v["shape"],
            )
            if v["type"] == "static":
                data = tf.tile(data, [meta["trajectory_length"], 1, 1])
            elif v["type"] == "dynamic_varlen":
                row_len = tf.reshape(
                    tf.io.decode_raw(features["length_" + k].values, tf.int32), [-1]
                )
                data = tf.RaggedTensor.from_row_lengths(data, row_lengths=row_len)
            outvar[k] = data
        return outvar

    @staticmethod
    def _cell_to_adj(cells) -> Tuple[List[int], List[int]]:
        """将网格单元转换为邻接表索引"""
        num_cells = np.shape(cells)[0]
        src = [cells[i][indx] for i in range(num_cells) for indx in [0, 1, 2]]
        dst = [cells[i][indx] for i in range(num_cells) for indx in [1, 2, 0]]
        return src, dst

    @staticmethod
    def _create_graph(src: List[int], dst: List[int]) -> DGLGraph:
        """根据源和目标节点索引创建 DGL 图"""
        graph = dgl.to_bidirected(dgl.graph((src, dst), idtype=torch.int32))
        return graph

    @staticmethod
    def _add_edge_features(graph: DGLGraph, pos: np.ndarray) -> DGLGraph:
        """计算并添加边特征（相对位移和距离）"""
        row, col = graph.edges()
        disp = torch.tensor(pos[row.long()] - pos[col.long()], dtype=torch.float)
        disp_norm = torch.linalg.norm(disp, dim=-1, keepdim=True)
        graph.edata["x"] = torch.cat((disp, disp_norm), dim=1)
        return graph

    @staticmethod
    def _normalize(invar: torch.Tensor, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """标准化张量"""
        return (invar - mu) / (std + 1e-8)

    @staticmethod
    def normalize_node(invar, mu, std):
        """标准化单个节点张量（供推理使用）"""
        if (invar.size()[-1] != mu.size()[-1]) or (invar.size()[-1] != std.size()[-1]):
            raise AssertionError("input and stats must have the same size")
        mu = mu.to(invar.device)
        std = std.to(invar.device)
        return (invar - mu.expand(invar.size())) / (std.expand(invar.size()) + 1e-8)

    @staticmethod
    def denormalize(invar, mu, std):
        """反标准化张量（供推理使用）"""
        mu = mu.to(invar.device)
        std = std.to(invar.device)
        denormalized_invar = invar * (std + 1e-8) + mu
        return denormalized_invar
    
    @staticmethod
    def _one_hot_encode(node_type: torch.Tensor) -> torch.Tensor:
        """对节点类型进行 One-Hot 编码"""
        node_type = torch.squeeze(node_type, dim=-1)
        node_type = torch.where(
            node_type == 0,
            torch.zeros_like(node_type),
            node_type - 3,
        )
        node_type = F.one_hot(node_type.long(), num_classes=4).float()
        return node_type

    @staticmethod
    def _drop_last(invar: np.ndarray) -> torch.Tensor:
        return torch.tensor(invar[0:-1], dtype=torch.float)

    @staticmethod
    def _push_forward(invar: np.ndarray) -> torch.Tensor:
        return torch.tensor(invar[1:], dtype=torch.float)

    @staticmethod
    def _push_forward_diff(invar: np.ndarray) -> torch.Tensor:
        return torch.tensor(invar[1:] - invar[0:-1], dtype=torch.float)

    @staticmethod
    def _get_rollout_mask(node_type: torch.Tensor) -> torch.Tensor:
        """生成 rollout 掩码"""
        mask = torch.logical_or(
            torch.eq(node_type, torch.zeros_like(node_type)),
            torch.eq(node_type, torch.zeros_like(node_type) + 5),
        )
        return mask

    @staticmethod
    def _add_noise(
        features: torch.Tensor, 
        targets: torch.Tensor, 
        noise_std: float, 
        noise_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """为训练数据添加高斯噪声"""
        noise = torch.normal(mean=0, std=noise_std, size=features.size())
        noise_mask = noise_mask.expand(features.size()[0], -1, 2)
        noise = torch.where(noise_mask, noise, torch.zeros_like(noise))
        features += noise
        targets -= noise
        return features, targets


# ==============================================================================
# DATAPIPE
# ==============================================================================

class DeepMind_CylinderFlowDatapipe:
    """
    为 DeepMind CylinderFlow (MeshGraphNet) 数据集创建 DataLoader [DGL 版本]
    """
    
    def __init__(self, params: Dict[str, Any], distributed: bool):
        self.params = params
        self.distributed = distributed
        
        # 创建分布式适配器
        self.dist = create_adapter()
        
        # 1. 初始化训练数据集
        if self.dist.is_rank0():
            logging.info("Initializing Train Dataset (DGL)...")
        self.train_dataset = DeepMind_CylinderFlowDataset(
            config=self.params, 
            mode='train'
        )
        
        # 2. 获取归一化统计数据
        self.stats = self.train_dataset.stats
        
        if self.distributed:
            self.dist.barrier()
            
        # 3. 初始化验证和测试数据集
        if self.dist.is_rank0():
            logging.info("Initializing Validation Dataset (DGL)...")
        self.val_dataset = DeepMind_CylinderFlowDataset(
            config=self.params, 
            mode='val', 
            stats=self.stats
        )
        
        if self.dist.is_rank0():
            logging.info("Initializing Test Dataset (DGL)...")
        self.test_dataset = DeepMind_CylinderFlowDataset(
            config=self.params, 
            mode='test', 
            stats=self.stats
        )

    def train_dataloader(self) -> Tuple[GraphDataLoader, Optional[DistributedSampler]]:
        # sampler = DistributedSampler(self.train_dataset, shuffle=True) if self.distributed else None
        
        if self.distributed:
            if self.dist.env_type == "megatron":
                dp_rank = mpu.get_data_parallel_rank()
                dp_size = mpu.get_data_parallel_world_size()
                sampler = DistributedSampler(
                    self.train_dataset,
                    num_replicas=dp_size,
                    rank=dp_rank,
                    shuffle=True
                )
            else:
                sampler = DistributedSampler(self.train_dataset, shuffle=True)
        else:
            sampler = None
        
        data_loader = GraphDataLoader(
            self.train_dataset,
            batch_size=self.params.dataloader.batch_size,
            drop_last=True if self.distributed else False,
            num_workers=self.params.dataloader.num_workers,
            pin_memory=True,
            shuffle=(sampler is None),
            sampler=sampler
        )
        return data_loader, sampler

    def val_dataloader(self) -> Tuple[GraphDataLoader, Optional[DistributedSampler]]:
        # sampler = DistributedSampler(self.val_dataset, shuffle=False) if self.distributed else None
        
        if self.distributed:
            if self.dist.env_type == "megatron":
                dp_rank = mpu.get_data_parallel_rank()
                dp_size = mpu.get_data_parallel_world_size()
                sampler = DistributedSampler(
                    self.val_dataset,
                    num_replicas=dp_size,
                    rank=dp_rank,
                    shuffle=False
                )
            else:
                sampler = DistributedSampler(self.val_dataset, shuffle=False)
        else:
            sampler = None

        data_loader = GraphDataLoader(
            self.val_dataset,
            batch_size=self.params.dataloader.batch_size,
            drop_last=True if self.distributed else False,
            num_workers=self.params.dataloader.num_workers,
            pin_memory=True,
            shuffle=False,
            sampler=sampler
        )
        return data_loader, sampler

    def test_dataloader(self) -> GraphDataLoader:
        sampler = None
        
        data_loader = GraphDataLoader(
            self.test_dataset,
            batch_size=self.params.dataloader.batch_size,
            drop_last=False,
            num_workers=self.params.dataloader.num_workers,
            pin_memory=True,
            shuffle=False,
            sampler=sampler
        )
        return data_loader