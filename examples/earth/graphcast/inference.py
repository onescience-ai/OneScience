import os
import sys

import numpy as np
import torch
from ruamel.yaml.scalarfloat import ScalarFloat

from onescience.datapipes.climate import ERA5HDF5Datapipe
from onescience.models.graphcast.graph_cast_net import GraphCastNet
from onescience.utils.fcn.YParams import YParams
from onescience.utils.graphcast.data_utils import StaticData

torch.serialization.add_safe_globals([ScalarFloat])


current_path = os.getcwd()
sys.path.append(current_path)

config_file_path = os.path.join(current_path, "conf/config.yaml")
cfg = YParams(config_file_path, "graphcast")
cfg["num_samples_per_year"] = 24
test_dataset = ERA5HDF5Datapipe(params=cfg, distributed=False, num_steps=1)
test_dataloader = test_dataset.test_dataloader()

ckpt = torch.load(
    f"{cfg.checkpoint_dir}/graphcast_finetune.pth",
    map_location="cuda:0",
    weights_only=True,
)
model_dtype = torch.bfloat16 if cfg.full_bf16 else torch.float32

input_dim_grid_nodes = (
    len(cfg.channels) + cfg.use_cos_zenith + 4 * cfg.use_time_of_year_index
) * (cfg.num_history + 1) + cfg.num_channels_static
graphcast_model = GraphCastNet(
    mesh_level=cfg.mesh_level,
    multimesh=cfg.multimesh,
    input_res=tuple(cfg.img_size),
    input_dim_grid_nodes=input_dim_grid_nodes,
    input_dim_mesh_nodes=3,
    input_dim_edges=4,
    output_dim_grid_nodes=len(cfg.channels),
    processor_type=cfg.processor_type,
    khop_neighbors=cfg.khop_neighbors,
    num_attention_heads=cfg.num_attention_heads,
    processor_layers=cfg.processor_layers,
    hidden_dim=cfg.hidden_dim,
    norm_type=cfg.norm_type,
    do_concat_trick=cfg.concat_trick,
    recompute_activation=cfg.recompute_activation,
)

graphcast_model.set_checkpoint_encoder(cfg.checkpoint_encoder)
graphcast_model.set_checkpoint_decoder(cfg.checkpoint_decoder)
graphcast_model = graphcast_model.to(dtype=model_dtype).to("cuda:0")
graphcast_model.load_state_dict(ckpt["model_state_dict"])

if hasattr(graphcast_model, "module"):
    latitudes = graphcast_model.module.latitudes
    longitudes = graphcast_model.module.longitudes
    lat_lon_grid = graphcast_model.module.lat_lon_grid
else:
    latitudes = graphcast_model.latitudes
    longitudes = graphcast_model.longitudes
    lat_lon_grid = graphcast_model.lat_lon_grid
static_data = (
    StaticData(cfg.static_dataset_path, latitudes, longitudes).get().to(device="cuda:0")
)
# ⚠️ 你的 checkpoint key
pred = []
label = []
# 4️⃣ 设置为 eval 模式
graphcast_model.eval()
with torch.no_grad():
    for j, data in enumerate(test_dataloader):
        invar = data[0].to(device="cuda:0")
        outvar = data[1].to(device="cuda:0")
        cos_zenith = data[2].to(device="cuda:0")
        in_idx = data[3].item()

        cos_zenith = torch.squeeze(cos_zenith, dim=2)
        cos_zenith = torch.clamp(cos_zenith, min=0.0) - 1.0 / torch.pi
        day_of_year, time_of_day = divmod(in_idx * cfg.dt, 24)
        normalized_day_of_year = torch.tensor(
            (day_of_year / 365) * (np.pi / 2), dtype=torch.float32, device="cuda:0"
        )
        normalized_time_of_day = torch.tensor(
            (time_of_day / (24 - cfg.dt)) * (np.pi / 2),
            dtype=torch.float32,
            device="cuda:0",
        )
        sin_day_of_year = torch.sin(normalized_day_of_year).expand(1, 1, 721, 1440)
        cos_day_of_year = torch.cos(normalized_day_of_year).expand(1, 1, 721, 1440)
        sin_time_of_day = torch.sin(normalized_time_of_day).expand(1, 1, 721, 1440)
        cos_time_of_day = torch.cos(normalized_time_of_day).expand(1, 1, 721, 1440)
        invar = torch.concat(
            (
                invar,
                cos_zenith,
                static_data,
                sin_day_of_year,
                cos_day_of_year,
                sin_time_of_day,
                cos_time_of_day,
            ),
            dim=1,
        )

        invar, outvar = invar.to(dtype=model_dtype), outvar.to(dtype=model_dtype)
        outvar_pred = graphcast_model(invar)

        print(f"infer process: {j+1}/{len(test_dataloader)}")
        pred.append(outvar_pred.float().cpu().numpy())
        label.append(outvar.float().cpu().numpy())

pred = np.concatenate(pred, axis=0)
label = np.concatenate(label, axis=0)
print(pred.shape, label.shape)
os.makedirs("result/", exist_ok=True)
np.save("result/pred.npy", pred)
np.save("result/label.npy", label)
