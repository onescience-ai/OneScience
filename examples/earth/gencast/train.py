import os
import sys
import numpy as np
import logging
import time
import pickle
import jax
import haiku as hk
import optax
import xarray
from onescience.models.gencast import gencast
from onescience.models.gencast.gencast import denoiser 
from onescience.models.gencast import normalization
from onescience.models.gencast import nan_cleaning
from onescience.models.gencast import xarray_jax
from onescience.models.gencast import xarray_tree

from onescience.datapipes.climate import ERA5Datapipe
from onescience.utils.YParams import YParams
from onescience.models.gencast.adapter import create_adapter


def main():

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger()

    ## Model config init
    config_file_path = os.path.join(current_path, "conf/config.yaml")
    cfg = YParams(config_file_path, "model")
    cfg_data = YParams(config_file_path, "datapipe")
    cfg_gencast = YParams(config_file_path, "gencast")

    ## DataLoader init
    datapipe = ERA5Datapipe(
        params=cfg_data, 
        distributed=False,
        input_steps=2,
        output_steps=1,
    )
    train_dataloader, train_sampler = datapipe.train_dataloader()
    val_dataloader, val_sampler = datapipe.val_dataloader()

    ## Adapter init (torch -> xarray 转换)
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
        target_shape=(181, 360)
    )

    ## Load normalization stats
    stats_dir = cfg_data.dataset.stats_dir
    mean_by_level = load_stats_to_xarray(
        stats_dir, "global_means.npy", cfg_data.dataset.channels,
        cfg_gencast.surface_variables, cfg_gencast.pressure_level_variables,
        cfg_gencast.pressure_levels, cfg_gencast.forcing_variables
    )
    stddev_by_level = load_stats_to_xarray(
        stats_dir, "global_stds.npy", cfg_data.dataset.channels,
        cfg_gencast.surface_variables, cfg_gencast.pressure_level_variables,
        cfg_gencast.pressure_levels, cfg_gencast.forcing_variables
    )
    diffs_stddev_by_level = load_stats_to_xarray(
        stats_dir, "global_diffs_stddev_by_level.npy", cfg_data.dataset.channels,
        cfg_gencast.surface_variables, cfg_gencast.pressure_level_variables,
        cfg_gencast.pressure_levels, cfg_gencast.forcing_variables
    )
    min_by_level = load_stats_to_xarray(
        stats_dir, "global_min_by_level.npy", cfg_data.dataset.channels,
        cfg_gencast.surface_variables, cfg_gencast.pressure_level_variables,
        cfg_gencast.pressure_levels, cfg_gencast.forcing_variables
    )

    ## Build GenCast model configs
    task_config = gencast.TaskConfig(
        input_variables=tuple(get_input_variables(cfg_gencast)),
        target_variables=tuple(get_target_variables(cfg_gencast)),
        forcing_variables=tuple(cfg_gencast.forcing_variables),
        pressure_levels=tuple(cfg_gencast.pressure_levels),
        input_duration=f"{cfg_data.dataset.time_res * 2}h",
    )
    
    sampler_config = gencast.SamplerConfig(
        max_noise_level=80.0,
        min_noise_level=0.03,
        num_noise_levels=20,
        rho=7.0,
        stochastic_churn_rate=2.5,
        churn_min_noise_level=0.75,
        churn_max_noise_level=float('inf'),
        noise_level_inflation_factor=1.05,
    )
    
    noise_config = gencast.NoiseConfig(
        training_noise_level_rho=7.0,
        training_max_noise_level=88.0,
        training_min_noise_level=0.02,
    )
    
    noise_encoder_config = denoiser.NoiseEncoderConfig(
        apply_log_first=True,
        base_period=16.0,
        num_frequencies=32,
        output_sizes=(32, 16),
    )
    
    sparse_transformer_config = denoiser.SparseTransformerConfig(
        attention_k_hop=16,
        d_model=512,
        num_layers=16,
        num_heads=4,
        attention_type='triblockdiag_mha',
        mask_type='full',
        block_q=1024,
        block_kv=512,
        block_kv_compute=256,
        block_q_dkv=512,
        block_kv_dkv=1024,
        block_kv_dkv_compute=1024,
        ffw_winit_final_mult=0.0,
        attn_winit_final_mult=0.0,
        ffw_hidden=2048,
        name=None,
    )
    
    denoiser_architecture_config = denoiser.DenoiserArchitectureConfig(
        sparse_transformer_config=sparse_transformer_config,
        mesh_size=5,
        latent_size=512,
        hidden_layers=1,
        radius_query_fraction_edge_length=0.6,
        norm_conditioning_features=('noise_level_encodings',),
        grid2mesh_aggregate_normalization=None,
        node_output_size=None,
    )

    ## Define model functions
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
    def loss_fn(inputs, targets, forcings):
        predictor = construct_wrapped_gencast()
        loss, diagnostics = predictor.loss(inputs, targets, forcings)
        return xarray_tree.map_structure(
            lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
            (loss, diagnostics),
        )

    def grads_fn(params, state, rng, inputs, targets, forcings):
        def _aux(params, state, rng, i, t, f):
            (loss, diagnostics), next_state = loss_fn.apply(params, state, rng, i, t, f)
            return loss, (diagnostics, next_state)
        (loss, (diagnostics, next_state)), grads = jax.value_and_grad(_aux, has_aux=True)(
            params, state, rng, inputs, targets, forcings
        )
        return loss, diagnostics, next_state, grads

    grads_fn_jitted = jax.jit(grads_fn)
    loss_fn_jitted = jax.jit(lambda params, state, rng, i, t, f: loss_fn.apply(params, state, rng, i, t, f)[0])

    ## Model init 
    logger.info("Initializing model parameters...")
    for data in train_dataloader:
        invar, outvar, time_index = data[0], data[1], data[4]
        sample_inputs, sample_targets, sample_forcings = adapter.convert(invar, outvar, time_index)
        break

    init_rng = jax.random.PRNGKey(42)
    params, state = loss_fn.init(
        rng=init_rng,
        inputs=sample_inputs,
        targets=sample_targets,
        forcings=sample_forcings,
    )

    ## Optimizer init
    learning_rate = getattr(cfg, 'learning_rate', 1e-4)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    ## Train process init
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    train_loss_file = f"{cfg.checkpoint_dir}/trloss.npy"
    valid_loss_file = f"{cfg.checkpoint_dir}/valoss.npy"
    best_valid_loss = 1.0e6
    best_loss_epoch = 0
    train_losses = np.empty((0,), dtype=np.float32)
    valid_losses = np.empty((0,), dtype=np.float32)
    rng = jax.random.PRNGKey(42)

    ## Get model params count
    total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print("\n\n")
    print("-" * 50)
    print(f"📂 Total params: {total_params:,}, {total_params / 1e6:.2f}M, {total_params / 1e9:.2f}B")
    print("-" * 50, "\n")

    ## Load model weight if there exist well-trained model
    checkpoint_path = f"{cfg.checkpoint_dir}/model_bak.pkl"
    if os.path.exists(checkpoint_path):
        print("\n\n")
        print("-" * 50)
        print(f"✅ Found checkpoint, loading and continue training...")
        print(f"If you want to train a new model, remove files in {cfg.checkpoint_dir}")
        print("-" * 50, "\n")
        ckpt = load_checkpoint(cfg.checkpoint_dir)
        params = ckpt["params"]
        state = ckpt["state"]
        opt_state = ckpt["opt_state"]
        best_valid_loss = ckpt["best_valid_loss"]
        best_loss_epoch = ckpt["best_loss_epoch"]
        if os.path.exists(train_loss_file):
            train_losses = np.load(train_loss_file)
        if os.path.exists(valid_loss_file):
            valid_losses = np.load(valid_loss_file)

    logger.info(f"Start training ...")

    for epoch in range(cfg.max_epoch):

        ## Training
        train_loss = 0
        start_time = time.time()
        for j, data in enumerate(train_dataloader):
            invar, outvar, time_index = data[0], data[1], data[4]
            inputs, targets, forcings = adapter.convert(invar, outvar, time_index)

            rng, step_rng = jax.random.split(rng)
            loss, diagnostics, state, grads = grads_fn_jitted(
                params, state, step_rng, inputs, targets, forcings
            )

            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

            train_loss += float(loss)
            logger.info(f'Train: Epoch {epoch}-{j+1}/{len(train_dataloader)} '
                        f'[cost {int((time.time()-start_time) // 60):02}:{int((time.time()-start_time) % 60):02}] '
                        f'[{(time.time()-start_time)/(j+1):.02f}s/{cfg_data.dataloader.batch_size}batch] '
                        f'loss:{train_loss / (j+1):.4f}')
            
        train_loss /= len(train_dataloader)

        ## Validation
        valid_loss = 0
        start_time = time.time()
        for j, data in enumerate(val_dataloader):
            invar, outvar, time_index = data[0], data[1], data[4]
            inputs, targets, forcings = adapter.convert(invar, outvar, time_index)

            rng, val_rng = jax.random.split(rng)
            loss, _ = loss_fn_jitted(params, state, val_rng, inputs, targets, forcings)

            loss_val = float(loss)

            valid_loss += loss_val
            logger.info(f'Valid: Epoch {epoch}-{j+1}/{len(val_dataloader)} '
                        f'[cost {int((time.time()-start_time) // 60):02}:{int((time.time()-start_time) % 60):02}] '
                        f'[{(time.time()-start_time)/(j+1):.02f}s/{cfg_data.dataloader.batch_size}batch] '
                        f'loss:{valid_loss / (j+1):.4f}')
            
        valid_loss /= len(val_dataloader)

        ## Save checkpoint
        is_save_ckp = False
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_loss_epoch = epoch
            save_checkpoint(
                params, state, opt_state, epoch, best_valid_loss, best_loss_epoch,
                cfg.checkpoint_dir,
                task_config, sampler_config, noise_config,
                noise_encoder_config, denoiser_architecture_config
            )
            is_save_ckp = True

        logger.info(f"Epoch [{epoch}/{cfg.max_epoch}], "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Valid Loss: {valid_loss:.4f}, "
                    f"Best loss at Epoch: {best_loss_epoch}"
                    + (", saving checkpoint" if is_save_ckp else ""))
        train_losses = np.append(train_losses, train_loss)
        valid_losses = np.append(valid_losses, valid_loss)
        np.save(train_loss_file, train_losses)
        np.save(valid_loss_file, valid_losses)

        ## Early stopping
        if epoch - best_loss_epoch > cfg.patience:
            print(f"Loss has not decreased in {cfg.patience} epochs, stopping training...")
            break

    logger.info("Training completed!")


def get_input_variables(cfg_gencast):
    variables = []
    for var in cfg_gencast.surface_variables:
        if var not in cfg_gencast.target_only_variables:
            variables.append(var)
    variables.extend(cfg_gencast.pressure_level_variables)
    variables.extend(cfg_gencast.forcing_variables)
    variables.extend(['geopotential_at_surface', 'land_sea_mask'])
    return variables


def get_target_variables(cfg_gencast):
    variables = list(cfg_gencast.surface_variables)
    variables.extend(cfg_gencast.pressure_level_variables)
    return variables


def load_stats_to_xarray(stats_dir, filename, channels, surface_vars, pressure_vars, pressure_levels, forcing_vars):
    data = np.load(os.path.join(stats_dir, filename)).squeeze()  
    data_vars = {}
    for var in surface_vars:
        if var in channels:
            idx = channels.index(var)
            data_vars[var] = xarray.DataArray(float(data[idx]), dims=[])
    for var in pressure_vars:
        var_data = []
        for level in pressure_levels:
            ch_name = f"{var}_{level}"
            if ch_name in channels:
                idx = channels.index(ch_name)
                var_data.append(float(data[idx]))
        if var_data:
            data_vars[var] = xarray.DataArray(
                np.array(var_data, dtype=np.float32),
                dims=['level'],
                coords={'level': list(pressure_levels)}
            )
    for var in forcing_vars:
        if var not in data_vars:
            data_vars[var] = xarray.DataArray(1.0, dims=[])
    for var in ['geopotential_at_surface', 'land_sea_mask']:
        if var not in data_vars:
            data_vars[var] = xarray.DataArray(1.0, dims=[])
    
    return xarray.Dataset(data_vars)


def save_checkpoint(params, state, opt_state, epoch, best_valid_loss, best_loss_epoch, 
                    checkpoint_dir, task_config, sampler_config, noise_config, 
                    noise_encoder_config, denoiser_architecture_config):
    os.makedirs(checkpoint_dir, exist_ok=True)
    def to_numpy(x):
        if hasattr(x, 'device'):
            return np.array(x)
        return x
    checkpoint = {
        "params": jax.tree_util.tree_map(to_numpy, params),
        "state": jax.tree_util.tree_map(to_numpy, state),
        "opt_state": jax.tree_util.tree_map(to_numpy, opt_state),
        "epoch": epoch,
        "best_valid_loss": best_valid_loss,
        "best_loss_epoch": best_loss_epoch,
        "task_config": task_config,
        "sampler_config": sampler_config,
        "noise_config": noise_config,
        "noise_encoder_config": noise_encoder_config,
        "denoiser_architecture_config": denoiser_architecture_config,
    }
    filepath = os.path.join(checkpoint_dir, "model.pkl")
    with open(filepath, 'wb') as f:
        pickle.dump(checkpoint, f)
    backup_path = os.path.join(checkpoint_dir, "model_bak.pkl")
    os.system(f"mv {filepath} {backup_path}")


def load_checkpoint(checkpoint_dir):
    filepath = os.path.join(checkpoint_dir, "model_bak.pkl")
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'rb') as f:
        checkpoint = pickle.load(f)
    return checkpoint


if __name__ == "__main__":
    current_path = os.getcwd()
    sys.path.append(current_path)
    main()