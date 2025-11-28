import random
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from bisect import bisect_right
from torch.utils.data import DataLoader, DistributedSampler
import copy
from tqdm import tqdm

from onescience.datapipes.core import BaseDataset
from onescience.distributed.manager import DistributedManager

def normalize_physics_props(case_params: Dict[str, float]):
    """
    Normalize the physics properties in-place.
    """
    density_mean = 5
    density_std = 4
    viscosity_mean = 0.00238
    viscosity_std = 0.005
    case_params["density"] = (
        case_params["density"] - density_mean
    ) / density_std
    case_params["viscosity"] = (
        case_params["viscosity"] - viscosity_mean
    ) / viscosity_std


def normalize_bc(case_params: Dict[str, float], key: str):
    """
    Normalize the boundary conditions in-place.
    """
    case_params[key] = case_params[key] / 50 - 0.5



def dump_json(data, path):
    with open(path, "w", encoding="utf8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path):
    """Load a JSON object from a file"""
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)

class CFDBenchDataset(BaseDataset):
    """
    CFDBench 数据集统一类
    支持 Tube, Cavity, Cylinder, Dam 四种场景
    支持 Auto (自回归) 和 Static (非自回归) 两种模式
    """
    
    DOMAIN = "cfd"
    DATA_FORMATS = ["npy", "json"]

    # 问题特定的元数据配置
    PROBLEM_CONFIG = {
        "tube": {
            "bc_key": "vel_in",
            "param_keys": ["vel_in", "density", "viscosity", "height", "width"],
            "data_dt": 0.1
        },
        "cavity": {
            "bc_key": "vel_top",
            "param_keys": ["vel_top", "density", "viscosity", "height", "width"],
            "data_dt": 0.1
        },
        "cylinder": {
            "bc_key": "vel_in",
            "param_keys": ["vel_in", "density", "viscosity", "height", "width", "center_x", "center_y", "radius"],
            "data_dt": 0.1 # 注意：Auto Dataset 原始代码是 0.001，但 Static 是 0.1，这里默认 0.1，可视情况调整
        },
        "dam": {
            "bc_key": "velocity",
            "param_keys": ["velocity", "density", "viscosity", "height", "width"],
            "data_dt": 0.1
        }
    }

    def __init__(self, config: Union[Dict[str, Any]], mode: str = 'train'):
        self.mode = mode
        super().__init__(config)
        
        # 解析配置
        self.problem_name = self.config.source.data_name.split("_")[0] 
        self.subset_name = self.config.source.data_name[len(self.problem_name) + 1 :]
        
        self.task_type = self.config.data.task_type # "auto" or "static"
        self.norm_props = self.config.data.norm_props
        self.norm_bc = self.config.data.norm_bc
        self.seed = self.config.data.seed
        
        # 获取当前问题的配置
        if self.problem_name not in self.PROBLEM_CONFIG:
            raise ValueError(f"Unknown problem: {self.problem_name}")
        self.curr_prob_cfg = self.PROBLEM_CONFIG[self.problem_name]
        
        # 初始化
        self._init_paths()
        self._load_metadata()
        self._init_data()

    def _init_paths(self):
        super()._init_paths() # 设置 self.data_path
        
        case_dirs = []
        search_keywords = ["prop", "bc", "geo"]
        problem_path = self.data_path / self.problem_name
        
        if not problem_path.exists():
             raise FileNotFoundError(f"Problem path not found: {problem_path}")

        for name in search_keywords:
            if name in self.subset_name:
                case_dir = problem_path / name
                if case_dir.exists():
                    # 按照 case 数字排序
                    this_case_dirs = sorted(
                        case_dir.glob("case*"), key=lambda x: int(x.name[4:])
                    )
                    case_dirs += this_case_dirs
        
        if not case_dirs:
            raise ValueError(f"No cases found for {self.problem_name} - {self.subset_name}")

        # 划分数据集
        random.seed(self.seed)
        random.shuffle(case_dirs)
        
        num_cases = len(case_dirs)
        ratios = self.config.data.split_ratios
        num_train = int(num_cases * ratios[0])
        num_val = int(num_cases * ratios[1])
        
        if self.mode == 'train':
            self.case_dirs = case_dirs[:num_train]
        elif self.mode == 'val':
            self.case_dirs = case_dirs[num_train : num_train + num_val]
        elif self.mode == 'test':
            self.case_dirs = case_dirs[num_train + num_val :]
        
        self.logger.info(f"[{self.mode}] Selected {len(self.case_dirs)} cases for {self.problem_name}.")

    def _load_metadata(self):
        pass

    def _init_data(self):
        """通用数据加载入口"""
        # 1. 选择 Raw Loader
        if self.problem_name == "tube":
            raw_loader = self._load_raw_tube
        elif self.problem_name == "cavity":
            raw_loader = self._load_raw_cavity
        elif self.problem_name == "cylinder":
            raw_loader = self._load_raw_cylinder
        elif self.problem_name == "dam":
            raw_loader = self._load_raw_dam
        else:
            raise NotImplementedError
            
        # 2. 根据任务类型加载数据
        if self.task_type == "auto":
            self._load_data_auto_generic(raw_loader)
        else:
            self._load_data_static_generic(raw_loader)

    # ================= Raw Loaders (Problem Specific) =================
    
    def _load_raw_tube(self, case_dir: Path):
        case_params = load_json(case_dir / "case.json")
        u = np.load(case_dir / "u.npy")
        v = np.load(case_dir / "v.npy")
        mask = np.ones_like(u)

        # Padding logic for Tube
        u = np.pad(u, ((0, 0), (0, 0), (1, 0)), mode="constant", constant_values=case_params["vel_in"])
        v = np.pad(v, ((0, 0), (0, 0), (1, 0)), mode="constant", constant_values=0)
        mask = np.pad(mask, ((0, 0), (0, 0), (1, 0)), mode="constant", constant_values=0)
        u = np.pad(u, ((0, 0), (1, 1), (0, 0)), mode="constant", constant_values=0)
        v = np.pad(v, ((0, 0), (1, 1), (0, 0)), mode="constant", constant_values=0)
        mask = np.pad(mask, ((0, 0), (1, 1), (0, 0)), mode="constant", constant_values=0)
        
        features = np.stack([u, v, mask], axis=1)
        return features, case_params

    def _load_raw_cavity(self, case_dir: Path):
        case_params = load_json(case_dir / "case.json")
        u = np.load(case_dir / "u.npy")
        v = np.load(case_dir / "v.npy")
        mask = np.ones_like(u)
        # Cavity logic: simple stack
        features = np.stack([u, v, mask], axis=1)
        return features, case_params

    def _load_raw_cylinder(self, case_dir: Path):
        case_params = load_json(case_dir / "case.json")
        u = np.load(case_dir / "u.npy")
        v = np.load(case_dir / "v.npy")
        mask = np.ones_like(u)
        
        # Geometry processing
        x_min, x_max = case_params.pop("x_min"), case_params.pop("x_max")
        y_min, y_max = case_params.pop("y_min"), case_params.pop("y_max")
        radius = case_params["radius"]
        case_params["center_x"] = -x_min
        case_params["center_y"] = -y_min
        
        height = y_max - y_min
        width = x_max - x_min
        case_params["height"] = height
        case_params["width"] = width

        dx = width / u.shape[2]
        dy = height / u.shape[1]
        
        # Create Mask (Circle) - vectorized logic for speed is possible but keeping loop for consistency
        # Vectorized version:
        Y, X = np.ogrid[:u.shape[1], :u.shape[2]]
        dist_sq = (x_min + X * dx - 0.5) ** 2 + (y_min + Y * dy - 0.5) ** 2
        mask[:, dist_sq <= radius**2] = 0

        # Padding
        u = np.pad(u, ((0, 0), (0, 0), (1, 0)), mode="constant", constant_values=case_params["vel_in"])
        v = np.pad(v, ((0, 0), (0, 0), (1, 0)), mode="constant", constant_values=0)
        mask = np.pad(mask, ((0, 0), (0, 0), (1, 0)), mode="constant", constant_values=0)
        u = np.pad(u, ((0, 0), (1, 1), (0, 0)), mode="constant", constant_values=0)
        v = np.pad(v, ((0, 0), (1, 1), (0, 0)), mode="constant", constant_values=0)
        mask = np.pad(mask, ((0, 0), (1, 1), (0, 0)), mode="constant", constant_values=0)
        
        features = np.stack([u, v, mask], axis=1)
        return features, case_params

    def _load_raw_dam(self, case_dir: Path):
        case_params = load_json(case_dir / "case.json")
        u = np.load(case_dir / "u.npy")
        v = np.load(case_dir / "v.npy")
        mask = np.ones_like(u)
        
        barrier_width = case_params["barrier_width"]
        barrier_height = case_params["barrier_height"]
        dx, dy = case_params["dx"], case_params["dy"]
        
        # Set barrier mask
        b_left, b_right = 0.5, 0.5 + barrier_width
        b_bot, b_top = 0, barrier_height
        
        idx_l, idx_r = int(b_left/dx), int(b_right/dx)
        idx_b, idx_t = int(b_bot/dy), int(b_top/dy)
        
        mask[:, idx_b:idx_t, idx_l:idx_r] = 0
        
        # Padding
        u = np.pad(u, ((0, 0), (0, 0), (1, 0)), mode="constant", constant_values=0)
        # Apply BC to left side padding
        # 注意：这里要计算 barrier_top 在 padding 后的索引，但在原始代码中直接用了 idx_t
        # 原始代码假定 barrier 也在左边界被填充了 velocity? 
        # 原始代码：u[:, :barrier_top_idx, :1] = case_params["velocity"]
        u[:, :idx_t, :1] = case_params["velocity"]
        
        v = np.pad(v, ((0, 0), (0, 0), (1, 0)), mode="constant", constant_values=0)
        mask = np.pad(mask, ((0, 0), (0, 0), (1, 0)), mode="constant", constant_values=0)
        
        # Top/Bottom padding
        u = np.pad(u, ((0, 0), (1, 1), (0, 0)), mode="constant", constant_values=0)
        v = np.pad(v, ((0, 0), (1, 1), (0, 0)), mode="constant", constant_values=0)
        mask = np.pad(mask, ((0, 0), (1, 1), (0, 0)), mode="constant", constant_values=0)
        
        features = np.stack([u, v, mask], axis=1)
        
        # Filter params
        wanted_keys = ["velocity", "density", "viscosity", "height", "width"]
        case_params = {k: case_params[k] for k in wanted_keys}
        return features, case_params

    # ================= Generic Loading Logic =================

    def _load_data_static_generic(self, raw_loader_func):
        """非自回归数据的通用加载逻辑"""
        self.case_params_list = []
        self.features = []
        self.num_frames = []
        
        param_keys = self.curr_prob_cfg["param_keys"]
        bc_key = self.curr_prob_cfg["bc_key"]

        self.logger.info(f"Loading {self.problem_name} STATIC data...")
        
        for case_dir in tqdm(self.case_dirs, desc="Loading Cases"):
            feats, params = raw_loader_func(case_dir)
            
            if self.norm_props:
                normalize_physics_props(params)
            if self.norm_bc:
                normalize_bc(params, bc_key)
                
            T = feats.shape[0]
            params_tensor = torch.tensor([params[k] for k in param_keys], dtype=torch.float32)
            
            self.case_params_list.append(params_tensor)
            self.features.append(torch.tensor(feats, dtype=torch.float32))
            self.num_frames.append(T)
            
        self.num_frames_before = [sum(self.num_frames[: i + 1]) for i in range(len(self.num_frames))]

    def _load_data_auto_generic(self, raw_loader_func):
        """自回归数据的通用加载逻辑"""
        self.delta_time = self.config.data.delta_time
        self.data_delta_time = self.curr_prob_cfg["data_dt"]
        self.time_step_size = int(self.delta_time / self.data_delta_time)
        
        bc_key = self.curr_prob_cfg["bc_key"]
        
        all_inputs = []
        all_labels = []
        self.case_params_list_auto = [] 
        self.case_ids = []
        
        self.logger.info(f"Loading {self.problem_name} AUTO data (dt={self.delta_time}, step={self.time_step_size})...")

        for case_id, case_dir in enumerate(tqdm(self.case_dirs, desc="Loading Cases")):
            feats, params = raw_loader_func(case_dir)
            
            if self.norm_props:
                normalize_physics_props(params)
            if self.norm_bc:
                normalize_bc(params, bc_key)
            
            # Split inputs/outputs
            inputs_raw = feats[:-self.time_step_size, :]
            outputs_raw = feats[self.time_step_size:, :]
            
            num_steps = len(outputs_raw)
            for i in range(num_steps):
                inp = torch.tensor(inputs_raw[i], dtype=torch.float32)
                out = torch.tensor(outputs_raw[i], dtype=torch.float32)
                
                # Convergence check (Standard for all CFDBench Auto datasets)
                # inp and out are (3, h, w), first 2 channels are u, v
                inp_magn = torch.sqrt(inp[0] ** 2 + inp[1] ** 2)
                out_magn = torch.sqrt(out[0] ** 2 + out[1] ** 2)
                # diff = torch.abs(inp_magn - out_magn).mean() 
                # 原始代码计算了diff但只做了assert valid check，这里保持一致
                
                if not torch.isnan(inp).any() and not torch.isnan(out).any():
                    all_inputs.append(inp)
                    all_labels.append(out)
                    self.case_ids.append(case_id)
            
            self.case_params_list_auto.append(params)
            
        self.inputs = torch.stack(all_inputs)
        self.labels = torch.stack(all_labels)

    # ================= Interface Implementation =================

    def __len__(self) -> int:
        if self.task_type == "auto":
            return len(self.inputs)
        else:
            return self.num_frames_before[-1] if self.num_frames_before else 0

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.task_type == "auto":
            inputs = self.inputs[idx] # (3, h, w) 
            label = self.labels[idx] # (3, h, w)
            case_id = self.case_ids[idx]
            c_params_raw = self.case_params_list_auto[case_id]
            
            return {
                "inputs": inputs,
                "label": label,
                "case_params": c_params_raw 
            }
        else:
            # Static
            case_id = bisect_right(self.num_frames_before, idx)
            if case_id == 0:
                frame_idx = idx
            else:
                frame_idx = idx - self.num_frames_before[case_id - 1]
            
            t = torch.tensor([frame_idx]).float()
            frame = self.features[case_id][frame_idx] # (3, h, w)
            case_params = self.case_params_list[case_id] # Tensor
            
            return {
                "case_params": case_params,
                "t": t,
                "label": frame 
            }

# --- Datapipe ---

class CFDBenchDatapipe:
    def __init__(self, config: Dict[str, Any], distributed: bool = False):
        self.config = config
        self.distributed = distributed
        
        self.train_dataset = CFDBenchDataset(copy.deepcopy(config), mode='train')
        self.val_dataset = CFDBenchDataset(copy.deepcopy(config), mode='val')
        self.test_dataset = CFDBenchDataset(copy.deepcopy(config), mode='test')

    def _get_collate_fn(self):
        task_type = self.config.data.task_type
        
        if task_type == "auto":
            def collate_fn_auto(batch):
                inputs_list = [b['inputs'] for b in batch]
                labels_list = [b['label'] for b in batch]
                case_params_list = [b['case_params'] for b in batch]
                
                inputs = torch.stack(inputs_list) 
                labels = torch.stack(labels_list)
                
                # 处理 Channel: Input=(u,v,mask), Label=(u,v,mask)
                # 训练时通常分离 mask
                mask = inputs[:, -1:] 
                inputs = inputs[:, :-1]
                labels = labels[:, :-1] 
                
                # Dict -> Tensor
                # 排除不需要的 key
                ignored_keys = ["rotated", "dx", "dy"]
                # 获取第一个样本的 key 作为基准
                keys = [k for k in case_params_list[0].keys() if k not in ignored_keys]
                vecs = [[cp[k] for k in keys] for cp in case_params_list]
                case_params = torch.tensor(vecs, dtype=torch.float32)
                
                return {
                    "inputs": inputs,
                    "label": labels,
                    "mask": mask,
                    "case_params": case_params
                }
            return collate_fn_auto
            
        else:
            def collate_fn_static(batch):
                case_params = torch.stack([b['case_params'] for b in batch])
                t = torch.stack([b['t'] for b in batch])
                label = torch.stack([b['label'] for b in batch])
                
                return {
                    "case_params": case_params,
                    "t": t,
                    "label": label
                }
            return collate_fn_static

    def train_dataloader(self):
        sampler = DistributedSampler(self.train_dataset) if self.distributed else None
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.dataloader.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.config.dataloader.num_workers,
            collate_fn=self._get_collate_fn(),
            drop_last=True
        ), sampler

    def val_dataloader(self):
        sampler = DistributedSampler(self.val_dataset, shuffle=False) if self.distributed else None
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.dataloader.eval_batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.config.dataloader.num_workers,
            collate_fn=self._get_collate_fn()
        ), sampler

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1, 
            shuffle=False,
            num_workers=self.config.dataloader.num_workers,
            collate_fn=self._get_collate_fn()
        )