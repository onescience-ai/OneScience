import sys
import os
import random
from functools import partial
import numpy as np
import torch.nn.functional as F

from onescience.datapipes.climate import ERA5Dataset
from onescience.utils.YParams import YParams
from onescience.models.pangu_distributed_4stage import Pangu,Pangu_stage0,Pangu_stage1,Pangu_stage2,Pangu_stage3
from onescience.metrics import L1_loss

import torch
import torch.distributed as dist

from onescience.distributed.megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from onescience.distributed.megatron.training import pretrain
from onescience.distributed.megatron.training import get_args
from onescience.distributed.megatron.core import mpu
from onescience.distributed.megatron.training.arguments import core_transformer_config_from_args

from onescience.distributed.pipelinetensorshapeconfig import PipelineTensorShapeConfig


def para_init():
    if torch.distributed.is_initialized() == False:
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    seed = 2222
    args = get_args()
    config = core_transformer_config_from_args(args)

    model_parallel_cuda_manual_seed(seed)
    torch.manual_seed(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return config

def train_valid_test_dataset_provider(train_val_test_num_samples):
    
    current_path = os.getcwd()
    sys.path.append(current_path)
    config_file_path = os.path.join(current_path, 'conf/config.yaml')
    cfg = YParams(config_file_path, 'model')
    cfg_data = YParams(config_file_path, "datapipe")
    cfg_data.world_size = 1

    train_dataset = ERA5Dataset(
        dataset_dir=cfg_data.dataset.data_dir,
        used_variables=cfg_data.dataset.channels,
        used_years=cfg_data.dataset.train_time,
        mode='train'
    )
    val_dataset = ERA5Dataset(
        dataset_dir=cfg_data.dataset.data_dir,
        used_variables=cfg_data.dataset.channels,
        used_years=cfg_data.dataset.val_time,
        mode='valid'
    )
    test_dataset = ERA5Dataset(
        dataset_dir=cfg_data.dataset.data_dir,
        used_variables=cfg_data.dataset.channels,
        used_years=cfg_data.dataset.test_time,
        mode='test'
    )

    print("len train:"+str(len(train_dataset)))
    print("len val:"+str(len(val_dataset)))
    print("len test:"+str(len(test_dataset)))
    return train_dataset, val_dataset, test_dataset

def build_pangu_model(cfg, cfg_data, config):
    pp_rank = mpu.get_pipeline_model_parallel_rank()

    stages = {
        0: Pangu_stage0,
        1: Pangu_stage1,
        2: Pangu_stage2,
        3: Pangu_stage3,
    }

    stage_cls = stages.get(pp_rank)
    if stage_cls is None:
        raise ValueError(f"Unsupported pipeline rank {pp_rank}")

    model = stage_cls(
        img_size=cfg_data.dataset.img_size,
        patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        window_size=cfg.window_size,
        config=config,
    )
    dim_size = cfg.embed_dim
    pp_config = PipelineTensorShapeConfig(
        num_stages=4,
        stage_shapes=[
            [[1, 521280, dim_size], [1, 521280, dim_size]],
            [[1, 131040, dim_size * 2], [1, 521280, dim_size]],
            [[1, 521280, dim_size], [1, 521280, dim_size]],
        ]
    )

    args = get_args()
    args.pipeline_tensor_shape_config = pp_config

    return model

def model_provider(pre_process=False, post_process=True):
    current_path = os.getcwd()
    
    config_file_path = os.path.join(current_path, 'conf/config.yaml')
    cfg = YParams(config_file_path, 'model')
    cfg_data = YParams(config_file_path, "datapipe")
    cfg_data.world_size = 1

    config = para_init()

    pangu_model = build_pangu_model(cfg, cfg_data, config)

    return pangu_model

def loss_func(x, y, weights, level_weight=1.0):
    return level_weight * (F.l1_loss(x, y, reduction='none') * weights).mean()

def loss_fun(out,outvar, filename):
    out_surface, out_upper_air = out
    
    tar_surface = outvar[:, :4, :, :].to("cuda", dtype=torch.float32)
    tar_upper_air = outvar[:, 4:, :, :].to("cuda", dtype=torch.float32)
    out_upper_air = out_upper_air.reshape(tar_upper_air.shape)    

    current_path = os.getcwd()
    local_rank = int(os.environ["LOCAL_RANK"])
    config_file_path = os.path.join(current_path, 'conf/config.yaml')
    cfg_data = YParams(config_file_path, "datapipe")
    surface_weights = torch.as_tensor(cfg_data.dataset.weights[:4], device=local_rank, dtype=torch.float32).view(1, -1, 1, 1)
    pressure_weights = torch.as_tensor(cfg_data.dataset.weights[4:], device=local_rank, dtype=torch.float32).view(1, -1, 1, 1)

    loss1 = loss_func(out_surface, tar_surface, surface_weights,  level_weight=0.25)
    loss2 = loss_func(out_upper_air, tar_upper_air, pressure_weights, level_weight=1.0)
    loss = loss1 + loss2

    filename = filename
    
    num_tokens = torch.tensor(1,device = "cuda")
    reporting_loss = torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])
    return  (loss, num_tokens, {'lm loss': reporting_loss})


def forward_step_func(data_iterator, model):
    current_path = os.getcwd()
    
    config_file_path = os.path.join(current_path, 'conf/config.yaml')
    cfg = YParams(config_file_path, 'model')
    cfg_data = YParams(config_file_path, "datapipe")
    cfg_data.world_size = 1
    
    land_mask = torch.from_numpy(np.load(os.path.join(cfg_data.dataset.static_dir, "land_mask.npy")).astype(np.float32))
    soil_type = torch.from_numpy(np.load(os.path.join(cfg_data.dataset.static_dir, "soil_type.npy")).astype(np.float32))
    topography = torch.from_numpy(np.load(os.path.join(cfg_data.dataset.static_dir, "topography.npy")).astype(np.float32))
    topography = (topography - topography.mean()) / (topography.std(unbiased=False) + 1e-6)
    surface_mask = torch.stack([land_mask, soil_type, topography], dim=0).cuda()
    surface_mask = surface_mask.unsqueeze(0).repeat(cfg_data.dataloader.batch_size, 1, 1, 1) 

    invar, outvar, _, _, fname = next(data_iterator)

    filename = fname[-1][0]
    
    invar_surface = invar[:, :4, :, :].to("cuda", dtype=torch.float32)
    invar_upper_air = invar[:, 4:, :, :].to("cuda", dtype=torch.float32)

    invar = torch.concat([invar_surface, surface_mask, invar_upper_air], dim=1)

    output = model(invar)

    loss_func = partial(loss_fun, outvar = outvar, filename = filename)
    return output,loss_func

if __name__=='__main__':
    train_valid_test_dataset_provider.is_distributed = True

    pretrain(
            train_valid_test_dataset_provider = train_valid_test_dataset_provider,
            model_provider = model_provider,
            model_type = None,
            forward_step_func = forward_step_func,
            args_defaults={'dataloader_type': 'cyclic'}
            )
