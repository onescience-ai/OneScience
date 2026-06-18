import os
import numpy as np
import glob
import torch
import sys
import h5py
from tqdm import tqdm

from onescience.utils.YParams import YParams
from onescience.datapipes.climate import ERA5Datapipe
from ruamel.yaml.scalarfloat import ScalarFloat
from onescience.modules.utils.graphcast.data_utils import StaticData
from onescience.modules.utils.graphcast.graph_utils import deg2rad
from onescience.models.graphcast.graph_cast_net import GraphCastNet


torch.serialization.add_safe_globals([ScalarFloat])


def get_stats(data_dir, channels):
    """从新版 h5 attrs 中读取变量列表，提取归一化参数"""
    h5_files = sorted(glob.glob(os.path.join(data_dir, "data", "*.h5")))
    with h5py.File(h5_files[0], "r") as f:
        ds = f["fields"]
        all_variables = [v.decode() if isinstance(v, bytes) else v for v in ds.attrs["variables"]]

    channel_indices = [all_variables.index(v) for v in channels]
    stats_dir = os.path.join(data_dir, "stats")
    mu = np.load(os.path.join(stats_dir, "global_means.npy"))   # [1, C, 1, 1]
    std = np.load(os.path.join(stats_dir, "global_stds.npy"))
    means = mu[:, channel_indices, :, :]
    stds = std[:, channel_indices, :, :]
    return means, stds


if __name__ == "__main__":
    current_path = os.getcwd()
    sys.path.append(current_path)

    ## Model config init
    config_file_path = os.path.join(current_path, "conf/config.yaml")
    cfg = YParams(config_file_path, "model")
    ## DataLoader init
    cfg_data = YParams(config_file_path, "datapipe")

    datapipe = ERA5Datapipe(
        dataset_dir=cfg_data.dataset.data_dir,
        used_variables=cfg_data.dataset.channels,
        used_years=cfg_data.dataset.test_time,
        distributed=False,
        batch_size=1,
        num_workers=4,
    )
    test_dataloader, _ = datapipe.get_dataloader("test")
    means, stds = get_stats(cfg_data.dataset.data_dir, cfg_data.dataset.channels)

    ckpt = torch.load(f"{cfg.checkpoint_dir}/model_finetune_bak.pth", map_location="cuda:0", weights_only=True)
    model_dtype = torch.bfloat16 if cfg.full_bf16 else torch.float32
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

    if hasattr(model, "module"):
        latitudes = model.module.latitudes
        longitudes = model.module.longitudes
    else:
        latitudes = model.latitudes
        longitudes = model.longitudes
    static_data = StaticData(cfg_data.dataset.static_dir, latitudes, longitudes).get().to(device="cuda:0")

    model.eval()
    os.makedirs('./result/output/', exist_ok=True)
    print(f"📂 samples will be generated to './result/output/'")
    with torch.no_grad():
        for data in tqdm(test_dataloader, desc="Inferring testset", unit="batch"):
            invar = data[0].to(device="cuda:0")
            cos_zenith = data[2].to(device="cuda:0")
            in_idx = data[3].item()
            filename = data[4][-1][0]

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

            np.save(f"result/output/{filename}.npy", pred_var)
