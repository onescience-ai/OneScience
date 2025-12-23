import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import logging
import functools
from typing import Any, Dict, Optional, Sequence, Union
import copy
from omegaconf import OmegaConf

from onescience.datapipes.core import BaseDataset
from onescience.distributed.manager import DistributedManager
from dgl.dataloading import GraphDataLoader

# DGL & TF Imports
try:
    import dgl
    from dgl.data import DGLDataset
except ImportError:
    raise ImportError("DGL is required for MeshGraphNet.")

try:
    import tensorflow.compat.v1 as tf
    tf.config.set_visible_devices([], "GPU")
except ImportError:
    raise ImportError("TensorFlow is required for reading .tfrecord files.")

# ============================================================
# [CRITICAL FIX] TFRecord Parsing Logic (Restored from Original)
# DeepMind datasets store data as serialized bytes, not raw types.
# ============================================================

# Feature Descriptions
_FEATURE_DESCRIPTION = {
    "position": tf.io.VarLenFeature(tf.string),
}

_FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT = _FEATURE_DESCRIPTION.copy()
_FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT["step_context"] = tf.io.VarLenFeature(tf.string)

_FEATURE_DTYPES = {
    "position": {"in": np.float32, "out": tf.float32},
    "step_context": {"in": np.float32, "out": tf.float32},
}

_CONTEXT_FEATURES = {
    "key": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    "particle_type": tf.io.VarLenFeature(tf.string), # Stored as string/bytes!
}

def convert_to_tensor(x, encoded_dtype):
    if len(x) == 1:
        out = np.frombuffer(x[0].numpy(), dtype=encoded_dtype)
    else:
        out = []
        for el in x:
            out.append(np.frombuffer(el.numpy(), dtype=encoded_dtype))
    out = tf.convert_to_tensor(np.array(out))
    return out

def parse_serialized_simulation_example(example_proto, metadata):
    """Parses a serialized simulation tf.SequenceExample."""
    if "context_mean" in metadata:
        feature_description = _FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT
    else:
        feature_description = _FEATURE_DESCRIPTION
        
    context, parsed_features = tf.io.parse_single_sequence_example(
        example_proto,
        context_features=_CONTEXT_FEATURES,
        sequence_features=feature_description,
    )
    
    # 1. Decode Sequence Features (Position, etc.)
    for feature_key, item in parsed_features.items():
        convert_fn = functools.partial(
            convert_to_tensor, encoded_dtype=_FEATURE_DTYPES[feature_key]["in"]
        )
        parsed_features[feature_key] = tf.py_function(
            convert_fn, inp=[item.values], Tout=_FEATURE_DTYPES[feature_key]["out"]
        )

    # 2. Reshape Position
    # There is an extra frame at the beginning
    position_shape = [metadata["sequence_length"] + 1, -1, metadata["dim"]]
    parsed_features["position"] = tf.reshape(
        parsed_features["position"], position_shape
    )
    
    # 3. Handle Context Features
    if "context_mean" in metadata:
        context_feat_len = len(metadata["context_mean"])
        sequence_length = metadata["sequence_length"] + 1
        parsed_features["step_context"] = tf.reshape(
            parsed_features["step_context"], [sequence_length, context_feat_len]
        )
        
    # 4. Decode Particle Type (Context)
    # This was the cause of the error: expecting int64 but getting string/bytes
    context["particle_type"] = tf.py_function(
        functools.partial(convert_to_tensor, encoded_dtype=np.int64),
        inp=[context["particle_type"].values],
        Tout=[tf.int64],
    )
    context["particle_type"] = tf.reshape(context["particle_type"], [-1])
    
    return context["particle_type"], parsed_features["position"] # Return Tuple

# ============================================================
# Graph Helpers
# ============================================================
def compute_edge_index(pos, radius):
    distances = torch.cdist(pos, pos, p=2)
    mask = distances < radius
    edge_index = torch.nonzero(mask).t().contiguous()
    return edge_index

def compute_edge_attr(graph, radius=0.015):
    edge_index = graph.edges()
    u, v = edge_index
    pos = graph.ndata["pos"]
    displacement = pos[v] - pos[u]
    distance = torch.norm(displacement, dim=-1, keepdim=True)
    dist_feat = torch.exp(-(distance**2) / radius**2)
    graph.edata["x"] = torch.cat((displacement, dist_feat), dim=-1)

def graph_update(graph, radius):
    num_edges = graph.num_edges()
    if num_edges > 0:
        graph.remove_edges(torch.arange(num_edges, device=graph.device))
    pos = graph.ndata["pos"]
    edge_index = compute_edge_index(pos, radius)
    graph.add_edges(edge_index[0], edge_index[1])
    compute_edge_attr(graph, radius)
    return graph

# ============================================================
# Dataset Class
# ============================================================
class DeepMindLagrangianDataset(BaseDataset): 
    DOMAIN = "cfd"
    DATA_FORMATS = ["tfrecord"]
    KINEMATIC_PARTICLE_ID = 3

    def __init__(self, config: Dict[str, Any], mode: str = 'train'):
        self.mode = mode
        self.dist = DistributedManager()
        
        super().__init__(config)
        
        self._init_paths()
        self.data_dir = self.data_path 
        
        self.global_cfg = self.config.get('global_data_cfg', {})
        
        self.num_history = self.global_cfg.get('num_history', 5)
        self.noise_std = self.global_cfg.get('noise_std', 0.0003)
        self.num_node_types = self.global_cfg.get('num_node_types', 6)
        
        self.split_name = self.config.get('split', mode)
        self.num_sequences = self.config.get('num_sequences', 1000)
        
        if self.dist.rank != 0:
            self.logger.setLevel(logging.WARNING)

        self._init_metadata()
        self._init_data()

    def _init_paths(self):
        super()._init_paths()
        
    def _init_metadata(self):
        meta_path = self.data_dir / "metadata.json"
        with open(meta_path, "r") as f:
            self.metadata = json.load(f)
            
        steps = self.config.get('num_steps', None)
        self.num_steps = steps if steps is not None else (self.metadata["sequence_length"] + 1)
        
        self.dt = self.metadata["dt"]
        self.radius = self.metadata["default_connectivity_radius"]
        self.bounds = self.metadata["bounds"][0]
        self.dim = self.metadata["dim"]
        
        self.vel_mean = torch.tensor(self.metadata["vel_mean"]).reshape(1, self.dim)
        self.vel_std = torch.tensor(self.metadata["vel_std"]).reshape(1, self.dim)
        self.acc_mean = torch.tensor(self.metadata["acc_mean"]).reshape(1, self.dim)
        self.acc_std = torch.tensor(self.metadata["acc_std"]).reshape(1, self.dim)

    def _init_data(self):
        if self.dist.rank == 0:
            self.logger.info(f"Loading TFRecords from {self.data_dir} ({self.split_name})...")
            
        dataset = tf.data.TFRecordDataset(str(self.data_dir / f"{self.split_name}.tfrecord"))
        
        # Use the RESTORED complex parse function
        dataset = dataset.map(
            functools.partial(parse_serialized_simulation_example, metadata=self.metadata)
        )
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        
        self.node_features = []
        self.node_type = []
        
        for i in range(self.num_sequences):
            try:
                # data_np is tuple (particle_type, position) from our custom parse function
                data_np = iterator.get_next()
            except tf.errors.OutOfRangeError:
                break
            
            # Note: particle_type is index 0, position is index 1
            typ = torch.from_numpy(data_np[0].numpy())
            pos = torch.from_numpy(data_np[1].numpy())[:self.num_steps]
            
            self.node_features.append({"position": pos})
            self.node_type.append(F.one_hot(typ, num_classes=self.num_node_types))
            
        self.num_samples_per_seq = self.num_steps - self.num_history - 1
        self.length = len(self.node_features) * self.num_samples_per_seq
        
        if self.dist.rank == 0:
            self.logger.info(f"Loaded {len(self.node_features)} sequences, total {self.length} samples.")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx):
        seq_idx, t_idx = divmod(idx, self.num_samples_per_seq)
        
        t = t_idx + self.num_history
        pos_seq = self.node_features[seq_idx]["position"]
        pos_window = pos_seq[t_idx : t + 2] 
        
        pos_t = pos_window[-2] 
        mask = self.node_type[seq_idx][:, self.KINEMATIC_PARTICLE_ID] == 0
        
        if self.mode == "train":
            noise = self._random_walk_noise(len(pos_window), pos_t.shape[0])
            noise = noise * mask.unsqueeze(-1) 
            pos_window = pos_window + noise
            
        vel = pos_window[1:] - pos_window[:-1] 
        acc = vel[1:] - vel[:-1] 
        
        vel = (vel - self.vel_mean) / self.vel_std
        acc = (acc - self.acc_mean) / self.acc_std
        
        vel_hist = vel[:-1].permute(1, 0, 2).flatten(1) 
        
        dist_bound = torch.cat([pos_t - self.bounds[0], self.bounds[1] - pos_t], dim=-1)
        feat_bound = torch.exp(-(dist_bound**2) / self.radius**2)
        feat_bound[dist_bound > self.radius] = 0
        
        x = torch.cat([pos_t, vel_hist, feat_bound, self.node_type[seq_idx]], dim=-1)
        y = torch.cat([pos_window[-1], vel[-1], acc[-1]], dim=-1)
        
        g = dgl.graph(([], []), num_nodes=pos_t.shape[0])
        g.ndata["x"] = x
        g.ndata["y"] = y
        g.ndata["pos"] = pos_t
        g.ndata["mask"] = mask
        
        g.ndata["t"] = torch.full((pos_t.shape[0],), t_idx, dtype=torch.long)
        
        g = graph_update(g, self.radius)
        return g

    def _random_walk_noise(self, num_steps, num_particles):
        num_vel = num_steps - 1
        std_step = self.noise_std / (num_vel ** 0.5)
        vel_noise = std_step * torch.randn(num_vel, num_particles, self.dim)
        vel_noise = vel_noise.cumsum(dim=0)
        pos_noise = torch.cat([torch.zeros(1, num_particles, self.dim), vel_noise.cumsum(dim=0)])
        pos_noise[-1] = pos_noise[-2] 
        return pos_noise
    
    def denormalize_velocity(self, velocity):
        return velocity * self.vel_std.to(velocity.device) + self.vel_mean.to(velocity.device)

    def denormalize_acceleration(self, acceleration):
        return acceleration * self.acc_std.to(acceleration.device) + self.acc_mean.to(acceleration.device)

    def unpack_targets(self, graph):
        ndata = graph.ndata["y"]
        pos = ndata[..., : self.dim]
        vel = ndata[..., self.dim : 2 * self.dim]
        acc = ndata[..., 2 * self.dim : 3 * self.dim]
        return pos, vel, acc

    def unpack_inputs(self, graph):
        ndata = graph.ndata["x"]
        pos = ndata[..., : self.dim]
        vel = ndata[..., self.dim : self.dim + self.dim * self.num_history]
        vel = vel.reshape(-1, self.num_history, self.dim).permute(1, 0, 2)
        node_type = ndata[..., -self.num_node_types :]
        return pos, vel, node_type

    def time_integrator(self, position, velocity, acceleration, dt, denormalize=True):
        if denormalize:
            velocity = self.denormalize_velocity(velocity)
            acceleration = self.denormalize_acceleration(acceleration)
        velocity_next = velocity + acceleration
        position_next = position + velocity_next
        return position_next, velocity_next

# ============================================================
# Datapipe Class
# ============================================================
class DeepMindLagrangianDatapipe:
    def __init__(self, cfg: Dict[str, Any], distributed: bool = False):
        self.cfg = cfg 
        self.distributed = distributed
        
        # Extract common configs
        # Use OmegaConf.to_container to resolve interpolations and convert to dict
        source_cfg = OmegaConf.to_container(cfg.datapipe.source, resolve=True)
        verbose = cfg.datapipe.get('verbose', False)
        
        global_data_cfg = {
            'data_dir': cfg.data.data_dir,
            'num_history': cfg.data.num_history,
            'noise_std': cfg.data.get('noise_std', 0.0003),
            'num_node_types': cfg.data.num_node_types
        }
        
        # Build split configs manually to ensure 'source' and 'verbose' are present
        
        # Train
        train_base = OmegaConf.to_container(cfg.data.train, resolve=True)
        train_cfg = {
            **train_base,
            'source': source_cfg,
            'verbose': verbose,
            'global_data_cfg': global_data_cfg
        }
        self.train_dataset = DeepMindLagrangianDataset(OmegaConf.create(train_cfg), mode='train')
        
        # Valid
        valid_base = OmegaConf.to_container(cfg.data.valid, resolve=True)
        valid_cfg = {
            **valid_base,
            'source': source_cfg,
            'verbose': verbose,
            'global_data_cfg': global_data_cfg
        }
        self.valid_dataset = DeepMindLagrangianDataset(OmegaConf.create(valid_cfg), mode='valid')
        
        # Test
        test_base = OmegaConf.to_container(cfg.data.test, resolve=True)
        test_cfg = {
            **test_base,
            'source': source_cfg,
            'verbose': verbose,
            'global_data_cfg': global_data_cfg
        }
        self.test_dataset = DeepMindLagrangianDataset(OmegaConf.create(test_cfg), mode='test')

    def train_dataloader(self):
        loader_cfg = self.cfg.datapipe.dataloader.train
        return GraphDataLoader(
            self.train_dataset,
            batch_size=loader_cfg.batch_size,
            shuffle=loader_cfg.shuffle, 
            num_workers=loader_cfg.num_workers,
            pin_memory=loader_cfg.pin_memory,
            drop_last=loader_cfg.drop_last,
            use_ddp=self.distributed
        )

    def val_dataloader(self):
        loader_cfg = self.cfg.datapipe.dataloader.valid
        return GraphDataLoader(
            self.valid_dataset,
            batch_size=loader_cfg.batch_size,
            shuffle=loader_cfg.shuffle, # 通常 False
            num_workers=loader_cfg.num_workers,
            pin_memory=loader_cfg.pin_memory,
            drop_last=loader_cfg.drop_last,
            use_ddp=self.distributed
        )
    
    def test_dataloader(self):
        loader_cfg = self.cfg.datapipe.dataloader.test
        return GraphDataLoader(
            self.test_dataset,
            batch_size=loader_cfg.batch_size, # 这里会是 1
            shuffle=loader_cfg.shuffle,
            num_workers=loader_cfg.num_workers,
            pin_memory=loader_cfg.pin_memory,
            drop_last=loader_cfg.drop_last,
            use_ddp=self.distributed
        )