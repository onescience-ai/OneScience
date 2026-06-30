import sys
import os
import random
from functools import partial
import numpy as np
import torch

from onescience.datapipes.climate import CMEMSHDF5Dataset
from onescience.utils.YParams import YParams
from onescience.models.xihe_distributed import Xihe_stage0, Xihe_stage1
from onescience.utils.fcn.darcy_loss import LpLoss

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
    cfg_data = YParams(config_file_path, "datapipe")
    cfg_data.world_size = 1

    train_dataset = CMEMSDataset(
        dataset_dir=cfg_data.dataset.data_dir,
        used_variables=cfg_data.dataset.channels,
        used_years=cfg_data.dataset.train_time,
        mode='train',
        output_steps=1,
        input_steps=1,
    )
    val_dataset = CMEMSDataset(
        dataset_dir=cfg_data.dataset.data_dir,
        used_variables=cfg_data.dataset.channels,
        used_years=cfg_data.dataset.val_time,
        mode='val',
        output_steps=1,
        input_steps=1,
    )
    test_dataset = CMEMSDataset(
        dataset_dir=cfg_data.dataset.data_dir,
        used_variables=cfg_data.dataset.channels,
        used_years=cfg_data.dataset.test_time,
        mode='test',
        output_steps=1,
        input_steps=1,
    )

    print("len train:" + str(len(train_dataset)))
    print("len val:" + str(len(val_dataset)))
    print("len test:" + str(len(test_dataset)))
    return train_dataset, val_dataset, test_dataset


def build_xihe_model(cfg, cfg_data, config):
    pp_rank = mpu.get_pipeline_model_parallel_rank()

    stages = {
        0: Xihe_stage0,
        1: Xihe_stage1,
    }

    stage_cls = stages.get(pp_rank)
    if stage_cls is None:
        raise ValueError(f"Unsupported pipeline rank {pp_rank}")

    mask_path = cfg.mask
    mask_path = os.path.expandvars(mask_path)

    model = stage_cls(
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        in_chans=cfg.in_chans,
        depth=cfg.depth,
        mask_full=mask_path,
        out_chans=cfg.out_chans,
        num_groups=cfg.num_groups,
        config=config,
    )

    import math
    H_out = math.ceil(cfg.img_size[0] / cfg.patch_size[0])
    W_out = math.ceil(cfg.img_size[1] / cfg.patch_size[1])
    N_full = H_out * W_out
    H_half = H_out // 2
    W_half = W_out // 2
    N_half = H_half * W_half
    dim_size = cfg.embed_dim

    pp_config = PipelineTensorShapeConfig(
        num_stages=2,
        stage_shapes=[
            [[1, N_half, dim_size * 2], [1, N_full, dim_size]],
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

    xihe_model = build_xihe_model(cfg, cfg_data, config)

    return xihe_model


def forward_step_func(data_iterator, model):
    data = next(data_iterator)
    invar = data[0].cuda().float()
    outvar = data[1].cuda().float()

    output = model(invar)

    def loss_func(output, outvar=outvar):
        loss_fn = LpLoss()
        loss = loss_fn(outvar, output)
        num_tokens = torch.tensor(1, device="cuda")
        reporting_loss = torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])
        return loss, num_tokens, {'lm loss': reporting_loss}

    return output, partial(loss_func, output)


if __name__ == '__main__':
    train_valid_test_dataset_provider.is_distributed = True

    pretrain(
        train_valid_test_dataset_provider=train_valid_test_dataset_provider,
        model_provider=model_provider,
        model_type=None,
        forward_step_func=forward_step_func,
        args_defaults={'dataloader_type': 'cyclic'}
    )
