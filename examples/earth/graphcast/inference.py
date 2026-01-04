import os
import numpy as np
import time
import torch
import sys
import logging
import h5py
import json
from tqdm import tqdm
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, LambdaLR
from torch.nn.parallel import DistributedDataParallel

from onescience.models.graphcast.graph_cast_net import GraphCastNet
from onescience.utils.graphcast.loss import GraphCastLossFunction
from onescience.utils.YParams import YParams
from onescience.launch.utils import load_checkpoint, save_checkpoint
from onescience.datapipes.climate import ERA5Datapipe
from onescience.utils.graphcast.data_utils import StaticData
from onescience.utils.graphcast.graph_utils import deg2rad
from ruamel.yaml.scalarfloat import ScalarFloat


torch.serialization.add_safe_globals([ScalarFloat])


def get_stats(cfg):
    meta_path = os.path.join(cfg.data_dir, 'metadata.json')
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    years = list(map(int, metadata["years"]))
    variables = metadata['variables']
    y = sorted(years)
    if cfg.train_ratio + cfg.val_ratio + cfg.test_ratio == 1:
        n_train = int(len(y) * cfg.train_ratio)
        n_val = int(len(y) * cfg.val_ratio)
        year_splits = {
            "train": y[:n_train],
            "val": y[n_train:n_train + n_val],
            "test": y[n_train + n_val:]
        }
    elif cfg.train_ratio + cfg.val_ratio + cfg.test_ratio == len(y):
        n_train =  cfg.train_ratio
        n_val = cfg.val_ratio
        year_splits = {
            "train": y[:n_train],
            "val": y[n_train:n_train + n_val],
            "test": y[n_train + n_val:]
        }
    else:
        print('\n\n')
        print('-' * 30)
        print('Train/Val/Test settings must use ratio or digital numbers')
        print('If using ratio, please ensure the sum of all ratios equal to 1')
        print(f'If using digital number, please ensure the sum of number equal to total years {len(y)}')
        print(f'❌❌ Now settings are {cfg.train_ratio}-{cfg.val_ratio}-{cfg.test_ratio}, please check.')
        print('-' * 30)
        print('\n\n')
        exit()

    selected_years = year_splits['test']
    total_files = []
    for year in selected_years:
        path = os.path.join(cfg.data_dir, 'data', str(year))
        files = sorted(os.listdir(path))
        samples_per_year = len(files) - 1
        total_files.extend(files[-samples_per_year:])
    
    channel_indices = [variables.index(v) for v in cfg.channels]
    mu = np.load(os.path.join(cfg.stats_dir, "global_means.npy"))  # shape: [1, M, 1, 1]
    std = np.load(os.path.join(cfg.stats_dir, "global_stds.npy"))
    means = mu[:, channel_indices, :, :]
    stds = std[:, channel_indices, :, :]
        
    return total_files, means, stds


if __name__ == "__main__":
    current_path = os.getcwd()
    sys.path.append(current_path)

    ## Model config init
    config_file_path = os.path.join(current_path, "conf/config.yaml")
    cfg = YParams(config_file_path, "model")
    ## DataLoader init
    cfg_data = YParams(config_file_path, "datapipe")
    cfg_data.dataloader.batch_size = 1
    test_dataset = ERA5Datapipe(params = cfg_data, distributed = False)
    test_dataloader = test_dataset.test_dataloader()
    total_files, means, stds = get_stats(cfg_data.dataset)

    ckpt = torch.load(f"{cfg.checkpoint_dir}/model_finetune_bak.pth", map_location="cuda:0", weights_only=True)
    model_dtype = torch.bfloat16 if cfg.full_bf16 else torch.float32
    ## DataLoader init
    input_dim_grid_nodes = (len(cfg_data.dataset.channels) + cfg.use_cos_zenith + 4 * cfg.use_time_of_year_index) * (cfg.num_history + 1) + cfg.num_channels_static
    model = GraphCastNet(mesh_level=cfg.mesh_level,
                         multimesh=cfg.multimesh,
                         input_res=tuple(cfg_data.dataset.img_size),
                         input_dim_grid_nodes=input_dim_grid_nodes,
                         input_dim_mesh_nodes=3,
                         input_dim_edges=4,
                         output_dim_grid_nodes=len(cfg_data.dataset.channels),
                         processor_type=cfg.processor_type,
                         khop_neighbors=cfg.khop_neighbors,
                         num_attention_heads=cfg.num_attention_heads,
                         processor_layers=cfg.processor_layers,
                         hidden_dim=cfg.hidden_dim,
                         norm_type=cfg.norm_type,
                         do_concat_trick=cfg.concat_trick,
                         recompute_activation=cfg.recompute_activation,
                         )

    model.set_checkpoint_encoder(cfg.checkpoint_encoder)
    model.set_checkpoint_decoder(cfg.checkpoint_decoder)
    model = model.to(dtype=model_dtype).to("cuda:0")
    model.load_state_dict(ckpt["model_state_dict"])

    model_dtype = torch.bfloat16 if cfg.full_bf16 else torch.float32
    model.set_checkpoint_encoder(cfg.checkpoint_encoder)
    model.set_checkpoint_decoder(cfg.checkpoint_decoder)
    model = model.to(dtype=model_dtype).to('cuda:0')
    if hasattr(model, "module"):
        latitudes = model.module.latitudes
        longitudes = model.module.longitudes
        lat_lon_grid = model.module.lat_lon_grid
    else:
        latitudes = model.latitudes
        longitudes = model.longitudes
        lat_lon_grid = model.lat_lon_grid
    static_data = StaticData(cfg_data.dataset.static_dir, latitudes, longitudes).get().to(device="cuda:0")

    # 4️⃣ 设置为 eval 模式
    model.eval()
    os.makedirs('result/output/', exist_ok=True)
    print(f"📂 Total {len(total_files)} samples will be generated to './result/output/'")
    with torch.no_grad():
        j = 0
        for data in tqdm(test_dataloader, desc="Inferring testset", unit="batch"):
            invar = data[0].to(device="cuda:0")
            cos_zenith = data[2].to(device="cuda:0")
            in_idx = data[3].item()

            cos_zenith = torch.squeeze(cos_zenith, dim=2)
            cos_zenith = torch.clamp(cos_zenith, min=0.0) - 1.0 / torch.pi
            day_of_year, time_of_day = divmod(in_idx * cfg.dt, 24)
            normalized_day_of_year = torch.tensor((day_of_year / 365) * (np.pi / 2), dtype=torch.float32, device="cuda:0")
            normalized_time_of_day = torch.tensor((time_of_day / (24 - cfg.dt)) * (np.pi / 2), dtype=torch.float32, device="cuda:0")
            sin_day_of_year = torch.sin(normalized_day_of_year).expand(1, 1, 721, 1440)
            cos_day_of_year = torch.cos(normalized_day_of_year).expand(1, 1, 721, 1440)
            sin_time_of_day = torch.sin(normalized_time_of_day).expand(1, 1, 721, 1440)
            cos_time_of_day = torch.cos(normalized_time_of_day).expand(1, 1, 721, 1440)
            invar = torch.concat((invar, cos_zenith, static_data, sin_day_of_year, cos_day_of_year, sin_time_of_day, cos_time_of_day), dim=1)

            invar = invar.to(dtype=model_dtype)
            pred_var = model(invar).to(dtype=torch.float32)
            pred_var = pred_var.cpu().numpy()
            pred_var = pred_var * stds + means
            
            np.save(f"result/output/{total_files[j][:-3]}.npy", pred_var)
            j += 1