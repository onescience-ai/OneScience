import os
import sys
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import xarray
from tqdm import tqdm

# GenCast 相关
from onescience.models.gencast import gencast
from onescience.models.gencast import denoiser
from onescience.models.gencast import normalization
from onescience.models.gencast import nan_cleaning


# 项目相关
current_path = os.getcwd()
sys.path.append(current_path)

from onescience.utils.YParams import YParams
from onescience.datapipes.climate import ERA5Datapipe
from onescience.models.gencast.adapter import create_adapter


def load_checkpoint(checkpoint_path):
    with open(checkpoint_path, 'rb') as f:
        ckpt = pickle.load(f)
    return ckpt


def load_normalization_stats(stats_dir, channels, surface_vars, pressure_vars, pressure_levels, forcing_vars):
    def load_to_xarray(filename):
        data = np.load(os.path.join(stats_dir, filename)).squeeze()
        data_vars = {}
        for var in surface_vars:
            if var in channels:
                data_vars[var] = xarray.DataArray(float(data[channels.index(var)]), dims=[])
        for var in pressure_vars:
            var_data = [float(data[channels.index(f"{var}_{l}")]) for l in pressure_levels if f"{var}_{l}" in channels]
            if var_data:
                data_vars[var] = xarray.DataArray(np.array(var_data, dtype=np.float32), dims=['level'], coords={'level': list(pressure_levels)})
        for var in forcing_vars + ['geopotential_at_surface', 'land_sea_mask']:
            if var not in data_vars:
                data_vars[var] = xarray.DataArray(1.0, dims=[])
        return xarray.Dataset(data_vars)
    
    return (load_to_xarray("global_means.npy"), 
            load_to_xarray("global_stds.npy"),
            load_to_xarray("global_diffs_stddev_by_level.npy"), 
            load_to_xarray("global_min_by_level.npy"))


def main():
    ## Config
    config_file_path = os.path.join(current_path, "conf/config.yaml")
    cfg = YParams(config_file_path, "model")
    cfg_data = YParams(config_file_path, "datapipe")
    cfg_gencast = YParams(config_file_path, "gencast")

    ## Load checkpoint
    checkpoint_path = os.path.join(cfg.checkpoint_dir, "model_bak.pkl")
    print(f"Loading checkpoint from {checkpoint_path}...")
    ckpt = load_checkpoint(checkpoint_path)
    
    params = ckpt["params"]
    state = ckpt["state"]
    task_config = ckpt["task_config"]
    sampler_config = ckpt["sampler_config"]
    noise_config = ckpt["noise_config"]
    noise_encoder_config = ckpt["noise_encoder_config"]
    denoiser_architecture_config = ckpt["denoiser_architecture_config"]
    print("Input variables:", len(task_config.input_variables), task_config.input_variables)
    print("\nTarget variables:", len(task_config.target_variables), task_config.target_variables)
    print(f"Loaded checkpoint: epoch {ckpt['epoch']}, best_loss {ckpt['best_valid_loss']:.4f}")

    ## Load normalization stats
    mean_by_level, stddev_by_level, diffs_stddev_by_level, min_by_level = load_normalization_stats(
        cfg_data.dataset.stats_dir, cfg_data.dataset.channels,
        cfg_gencast.surface_variables, cfg_gencast.pressure_level_variables,
        cfg_gencast.pressure_levels, cfg_gencast.forcing_variables
    )

    ## Build model
    def construct_wrapped_gencast():
        predictor = gencast.GenCast(
            sampler_config=sampler_config,
            task_config=task_config,
            denoiser_architecture_config=denoiser_architecture_config,
            noise_config=noise_config,
            noise_encoder_config=noise_encoder_config,
        )
        predictor = normalization.InputsAndResiduals(
            predictor,
            diffs_stddev_by_level=diffs_stddev_by_level,
            mean_by_level=mean_by_level,
            stddev_by_level=stddev_by_level,
        )
        predictor = nan_cleaning.NaNCleaner(
            predictor=predictor,
            reintroduce_nans=True,
            fill_value=min_by_level,
            var_to_clean='sea_surface_temperature',
        )
        return predictor

    @hk.transform_with_state
    def run_forward(inputs, targets_template, forcings):
        predictor = construct_wrapped_gencast()
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    run_forward_jitted = jax.jit(
        lambda rng, i, t, f: run_forward.apply(params, state, rng, i, t, f)[0]
    )

    ## DataLoader
    cfg_data.dataloader.batch_size = 1
    datapipe = ERA5Datapipe(params=cfg_data, distributed=False, input_steps=2, output_steps=1)
    test_dataloader = datapipe.test_dataloader()

    ## Adapter
    adapter = create_adapter(
        channels=cfg_data.dataset.channels,
        surface_variables=cfg_gencast.surface_variables,
        pressure_level_variables=cfg_gencast.pressure_level_variables,
        pressure_levels=cfg_gencast.pressure_levels,
        target_only_variables=cfg_gencast.target_only_variables,
        static_dir=cfg_data.dataset.static_dir,
        static_files=cfg_gencast.static_files,
        img_size=cfg_data.dataset.img_size,
        time_interval_hours=cfg_data.dataset.time_res,
        target_shape=(181, 360),
    )

    ## Inference
    os.makedirs('result/output/', exist_ok=True)
    rng = jax.random.PRNGKey(42)

    print(f"Starting inference...")
    for j, data in enumerate(tqdm(test_dataloader, desc="Inference")):
        invar, outvar, time_index = data[0], data[1], data[4]
        inputs, targets, forcings = adapter.convert(invar, outvar, time_index)
  
        # 推理
        rng, step_rng = jax.random.split(rng)
        predictions = run_forward_jitted(step_rng, inputs, targets * jnp.nan, forcings)
        
        # 获取输出时间戳作为文件名
        # time_index 格式: [['t-1'], ['t'], ['t+1']] 或 ['t-1', 't', 't+1']
        if isinstance(time_index[0], (list, tuple)):
            output_timestamp = time_index[-1][0]  # 最后一个是输出时刻
        else:
            output_timestamp = time_index[-1]
        
        # 将 predictions 转换为 [B, C, H, W] 格式
        pred_list = []
        
        # Surface 变量
        for var in cfg_gencast.surface_variables:
            if var in predictions.data_vars:
                # [batch, time, lat, lon] -> [batch, lat, lon]
                pred_list.append(predictions[var].values[:, 0, :, :])
        
        # Pressure level 变量
        for var in cfg_gencast.pressure_level_variables:
            if var in predictions.data_vars:
                # [batch, time, level, lat, lon] -> [batch, level, lat, lon]
                data = predictions[var].values[:, 0, :, :, :]
                for level_idx in range(data.shape[1]):
                    pred_list.append(data[:, level_idx, :, :])
        
            # 拼接为 [B, C, H, W]
        pred_var = np.stack(pred_list, axis=1)
        # 保存，文件名使用输出时间戳
        np.save(f"result/output/{output_timestamp}.npy", pred_var)

    print("Inference completed!")


if __name__ == "__main__":
    main()