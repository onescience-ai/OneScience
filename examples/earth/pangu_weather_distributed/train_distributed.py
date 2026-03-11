import sys
import os
import random
from functools import partial
import numpy as np

from onescience.datapipes.climate import ERA5Dataset
from onescience.utils.YParams import YParams
from onescience.models.pangu_distributed import Pangu,Pangu_stage0,Pangu_stage1
from onescience.metrics import L1_loss

import torch
import torch.distributed as dist

from onescience.distributed.megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from onescience.distributed.megatron.training import pretrain
from onescience.distributed.megatron.training import get_args
from onescience.distributed.megatron.core import mpu
from onescience.distributed.megatron.training.arguments import core_transformer_config_from_args


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
    # train_num, val_num, test_num = train_val_test_num_samples
    
    current_path = os.getcwd()
    sys.path.append(current_path)
    config_file_path = os.path.join(current_path, 'conf/config.yaml')
    cfg = YParams(config_file_path, 'pangu')
    cfg.world_size = 1
    cfg.stats_dir = "/public/onestore/onedatasets/ERA5/newh5/stats/" 
    cfg.static_dir =  "/public/onestore/onedatasets/ERA5/newh5/static/"
    cfg.data_dir =  "/public/onestore/onedatasets/ERA5/newh5" 

    train_dataset = ERA5Dataset(params=cfg, mode='train')
    val_dataset = ERA5Dataset(params=cfg, mode='val')
    test_dataset = ERA5Dataset(params=cfg, mode='test')

    print("len train:"+str(len(train_dataset)))
    print("len val:"+str(len(val_dataset)))
    print("len test:"+str(len(test_dataset)))
    return train_dataset, val_dataset, test_dataset

def model_provider(pre_process=False, post_process=True):
    current_path = os.getcwd()
    
    config_file_path = os.path.join(current_path, 'conf/config.yaml')
    cfg = YParams(config_file_path, 'pangu')
    cfg.world_size = 1

    config = para_init()
    
    ppworldsize = mpu.get_pipeline_model_parallel_world_size()
    pprank = mpu.get_pipeline_model_parallel_rank()
    if ppworldsize == 2:
        if pprank == 0: 
            pangu_model = Pangu_stage0(img_size=cfg.img_size,
                            patch_size=cfg.patch_size,
                            embed_dim=cfg.embed_dim,
                            num_heads=cfg.num_heads,
                            window_size=cfg.window_size,
                            config = config)
        if pprank == 1: 
            pangu_model = Pangu_stage1(img_size=cfg.img_size,
                            patch_size=cfg.patch_size,
                            embed_dim=cfg.embed_dim,
                            num_heads=cfg.num_heads,
                            window_size=cfg.window_size,
                            config = config)
    elif ppworldsize == 1:
           pangu_model = Pangu(img_size=cfg.img_size,
                            patch_size=cfg.patch_size,
                            embed_dim=cfg.embed_dim,
                            num_heads=cfg.num_heads,
                            window_size=cfg.window_size,
                            config = config)
    return pangu_model

def loss_func(x, y):
    return L1_loss(x, y)

def loss_fun(out,outvar):
    out_surface, out_upper_air = out
    
    tar_surface = outvar[:, :4, :, :].to("cuda", dtype=torch.float32)
    tar_upper_air = outvar[:, 4:, :, :].to("cuda", dtype=torch.float32)

    out_upper_air = out_upper_air.reshape(tar_upper_air.shape)    
    loss1 = loss_func(tar_surface, out_surface)
    loss2 = loss_func(tar_upper_air, out_upper_air)
    loss = loss1 * 0.25 + loss2
    num_tokens = torch.tensor(1,device = "cuda")
    reporting_loss = torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])
    return  (loss, num_tokens, {'lm loss': reporting_loss})


def forward_step_func(data_iterator, model):
    current_path = os.getcwd()
    
    config_file_path = os.path.join(current_path, 'conf/config.yaml')
    cfg = YParams(config_file_path, 'pangu')
    cfg.world_size = 1
    
    land_mask = torch.from_numpy(np.load(os.path.join(cfg.mask_dir, "land_mask.npy")).astype(np.float32))
    soil_type = torch.from_numpy(np.load(os.path.join(cfg.mask_dir, "soil_type.npy")).astype(np.float32))
    topography = torch.from_numpy(np.load(os.path.join(cfg.mask_dir, "topography.npy")).astype(np.float32))
    surface_mask = torch.stack([land_mask, soil_type, topography], dim=0).cuda()
    surface_mask = surface_mask.unsqueeze(0).repeat(cfg.batch_size, 1, 1, 1)

    invar, outvar, _, _ = next(data_iterator)
    
    invar_surface = invar[:, :4, :, :].to("cuda", dtype=torch.float32)
    invar_upper_air = invar[:, 4:, :, :].to("cuda", dtype=torch.float32)

    invar = torch.concat([invar_surface, surface_mask, invar_upper_air], dim=1)

    output = model(invar)

    loss_func = partial(loss_fun, outvar = outvar)
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
