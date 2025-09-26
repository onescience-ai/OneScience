import argparse
import dataclasses
import functools
import math
from typing import Optional

import cartopy.crs as ccrs  # 调用cartopy.crs模块，用于创建投影
# 调用cartopy.feature模块，引入默认地理信息
import cartopy.feature as cfeature
import haiku as hk
import jax
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray

from onescience.flax_models.graphcast import (
    autoregressive,
    casting,
    checkpoint,
    data_utils,
    graphcast,
    normalization,
    rollout,
    xarray_jax,
    xarray_tree,
)


def argsparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained", type=str, default="./", help="pretrained model")
    parser.add_argument(
        "--dataset", type=str, help="input data")
    parser.add_argument(
        "--mode", type=str, help="the ways of getting model params")
    parser.add_argument(
        "--var",
        type=str,
        default="2m_temperature",
        help="visualizing atmospheric variables",
    )
    parser.add_argument(
        "--level", type=int, default=500, help="atmospheric pressure level"
    )
    return parser


def print_arguments(args):
    print("-----------  Running Arguments -----------")
    for arg, value in sorted(vars(args).items()):
        print("%s: %s" % (arg, value))
    print("------------------------------------------")


def parse_file_parts(file_name):
    return dict(part.split("-", 1) for part in file_name.split("_"))


def select(
    data: xarray.Dataset,
    variable: str,
    level: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> xarray.Dataset:
    data = data[variable]
    if "batch" in data.dims:
        data = data.isel(batch=0)
    if (
        max_steps is not None
        and "time" in data.sizes
        and max_steps < data.sizes["time"]
    ):
        data = data.isel(time=range(0, max_steps))
    if level is not None and "level" in data.coords:
        data = data.sel(level=level)
    return data


def scale(
    data: xarray.Dataset,
    center: Optional[float] = None,
    robust: bool = False,
) -> tuple[xarray.Dataset, matplotlib.colors.Normalize, str]:
    vmin = np.nanpercentile(data, (2 if robust else 0))
    vmax = np.nanpercentile(data, (98 if robust else 100))
    if center is not None:
        diff = max(vmax - center, center - vmin)
        vmin = center - diff
        vmax = center + diff
    return (
        data,
        matplotlib.colors.Normalize(vmin, vmax),
        ("RdBu_r" if center is not None else "viridis"),
    )


def plot_data(
    data: dict[str, xarray.Dataset],
    fig_title: str,
    plot_size: float = 5,
    robust: bool = False,
    cols: int = 4,
) -> tuple[xarray.Dataset, matplotlib.colors.Normalize, str]:

    first_data = next(iter(data.values()))[0]
    max_steps = first_data.sizes.get("time", 1)
    assert all(max_steps == d.sizes.get("time", 1)
               for d, _, _ in data.values())

    cols = min(cols, len(data))
    rows = math.ceil(len(data) / cols)
    figure = plt.figure(
        figsize=(plot_size * 2 * cols, plot_size * rows))
    figure.suptitle(fig_title, fontsize=16)
    figure.subplots_adjust(wspace=0, hspace=0)
    figure.tight_layout()

    images = []
    for i, (title, (plot_data, norm, cmap)) in enumerate(data.items()):
        ax = figure.add_subplot(
            rows, cols, i + 1, projection=ccrs.PlateCarree())
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)

        # 设置地图范围（中国区域）
        ax.set_extent([73, 135, 18, 54])

        # 绘制中国地图的边界和其他地理特征
        ax.add_feature(
            cfeature.LAND, edgecolor="black")  # 陆地
        ax.add_feature(cfeature.COASTLINE,
                       edgecolor="black")  # 海岸线
        ax.add_feature(cfeature.BORDERS,
                       edgecolor="gray")  # 国家边界
        # ax.add_feature(cfeature.LAND, facecolor='lightgray')  # 填充陆地颜色

        im = ax.imshow(
            plot_data.isel(time=0, missing_dims="ignore"),
            norm=norm,
            origin="lower",
            cmap=cmap,
        )
        plt.colorbar(
            mappable=im,
            ax=ax,
            orientation="vertical",
            pad=0.02,
            aspect=16,
            shrink=0.75,
            cmap=cmap,
            extend=("both" if robust else "neither"),
        )
        plt.savefig(f"{fig_title}.png")
        images.append(im)


def save_var_diff(
    eval_targets, predictions, plot_pred_variable, plot_pred_level, plot_max_steps=1
):
    plot_size = 5
    plot_max_steps = min(predictions.sizes["time"], 1)

    data = {
        "Targets": scale(
            select(eval_targets, plot_pred_variable,
                   plot_pred_level, plot_max_steps),
            robust=True,
        ),
        "Predictions": scale(
            select(predictions, plot_pred_variable,
                   plot_pred_level, plot_max_steps),
            robust=True,
        ),
        "Diff": scale(
            (
                select(
                    eval_targets, plot_pred_variable, plot_pred_level, plot_max_steps
                )
                - select(
                    predictions, plot_pred_variable, plot_pred_level, plot_max_steps
                )
            ),
            robust=True,
            center=0,
        ),
    }
    fig_title = plot_pred_variable
    if "level" in predictions[plot_pred_variable].coords:
        fig_title += f"_at_{plot_pred_level}_hPa"

    plot_data(data, fig_title, plot_size, True)


def data_valid_for_model(
    file_name: str,
    model_config: graphcast.ModelConfig,
    task_config: graphcast.TaskConfig,
):
    file_parts = parse_file_parts(
        file_name.removesuffix(".nc"))
    return (
        model_config.resolution in (
            0, float(file_parts["res"]))
        and len(task_config.pressure_levels) == int(file_parts["levels"])
        and (
            (
                "total_precipitation_6hr" in task_config.input_variables
                and file_parts["source"] in ("era5", "fake")
            )
            or (
                "total_precipitation_6hr" not in task_config.input_variables
                and file_parts["source"] in ("hres", "fake")
            )
        )
    )


def load_data():
    # Load normalization data
    with open("./stats/stats_diffs_stddev_by_level.nc", "rb") as f:
        diffs_stddev_by_level = xarray.load_dataset(
            f).compute()
    with open("./stats/stats_mean_by_level.nc", "rb") as f:
        mean_by_level = xarray.load_dataset(f).compute()
    with open("./stats/stats_stddev_by_level.nc", "rb") as f:
        stddev_by_level = xarray.load_dataset(f).compute()
    return diffs_stddev_by_level, mean_by_level, stddev_by_level


# Build jitted functions, and possibly initialize random weights
def construct_wrapped_graphcast(
    model_config: graphcast.ModelConfig, task_config: graphcast.TaskConfig
):
    """Constructs and wraps the GraphCast Predictor."""
    # Deeper one-step predictor.
    predictor = graphcast.GraphCast(
        model_config, task_config)

    # Modify inputs/outputs to `graphcast.GraphCast` to handle conversion to
    # from/to float32 to/from BFloat16.
    predictor = casting.Bfloat16Cast(predictor)

    # Modify inputs/outputs to `casting.Bfloat16Cast` so the casting to/from
    # BFloat16 happens after applying normalization to the inputs/targets.
    # 加载标准化数据
    diffs_stddev_by_level, mean_by_level, stddev_by_level = load_data()
    predictor = normalization.InputsAndResiduals(
        predictor,
        diffs_stddev_by_level=diffs_stddev_by_level,
        mean_by_level=mean_by_level,
        stddev_by_level=stddev_by_level,
    )

    # Wraps everything so the one-step model can produce trajectories.
    predictor = autoregressive.Predictor(
        predictor, gradient_checkpointing=True)
    return predictor


@hk.transform_with_state
def run_forward(model_config, task_config, inputs, targets_template, forcings):
    predictor = construct_wrapped_graphcast(
        model_config, task_config)
    return predictor(inputs, targets_template=targets_template, forcings=forcings)


@hk.transform_with_state
def loss_fn(model_config, task_config, inputs, targets, forcings):

    predictor = construct_wrapped_graphcast(
        model_config, task_config)
    loss, diagnostics = predictor.loss(
        inputs, targets, forcings)
    return xarray_tree.map_structure(
        lambda x: xarray_jax.unwrap_data(
            x.mean(), require_jax=True),
        (loss, diagnostics),
    )


def grads_fn(params, state, model_config, task_config, inputs, targets, forcings):

    def _aux(params, state, i, t, f):
        (loss, diagnostics), next_state = loss_fn.apply(
            params, state, jax.random.PRNGKey(
                0), model_config, task_config, i, t, f
        )
        return loss, (diagnostics, next_state)

    (loss, (diagnostics, next_state)), grads = jax.value_and_grad(_aux, has_aux=True)(
        params, state, inputs, targets, forcings
    )
    return loss, diagnostics, next_state, grads


# Jax doesn't seem to like passing configs as args through the jit. Passing it
# in via partial (instead of capture by closure) forces jax to invalidate the
# jit cache if you change configs.


def main():
    # 打印参数配置
    parser = argsparser()
    FLAGS = parser.parse_args()
    print_arguments(FLAGS)

    # 加载预训练模型
    source = FLAGS.mode
    if source == "Random":
        params = None  # Filled in below
        state = {}
        model_config = graphcast.ModelConfig(
            resolution=0,
            mesh_size=4,
            latent_size=32,
            gnn_msg_steps=1,
            hidden_layers=1,
            radius_query_fraction_edge_length=0.6,
        )
        task_config = graphcast.TaskConfig(
            input_variables=graphcast.TASK.input_variables,
            target_variables=graphcast.TASK.target_variables,
            forcing_variables=graphcast.TASK.forcing_variables,
            pressure_levels=graphcast.PRESSURE_LEVELS[13],
            input_duration=graphcast.TASK.input_duration,
        )
    else:
        assert source == "Checkpoint"
        with open(FLAGS.pretrained, "rb") as f:
            ckpt = checkpoint.load(f, graphcast.CheckPoint)
        params = ckpt.params
        state = {}

        model_config = ckpt.model_config
        task_config = ckpt.task_config
        print("Model description:\n",
              ckpt.description, "\n")
        print("Model license:\n", ckpt.license, "\n")
        print(model_config)
        print(task_config)

    # 加载推理和训练数据
    dataset_file = FLAGS.dataset
    with open(dataset_file, "rb") as f:
        example_batch = xarray.load_dataset(f).compute()
        # example_batch = xarray.concat([example_batch,example_batch,example_batch,example_batch], dim="batch")
    # 2 for input, >=1 for targets
    assert example_batch.sizes["time"] >= 3
    print(
        ", ".join(
            [
                f"{k}: {v}"
                for k, v in parse_file_parts(dataset_file.removesuffix(".nc")).items()
            ]
        )
    )

    train_inputs, train_targets, train_forcings = (
        data_utils.extract_inputs_targets_forcings(
            example_batch,
            target_lead_times=slice("6h", f"{1 * 6}h"),
            **dataclasses.asdict(task_config),
        )
    )

    eval_inputs, eval_targets, eval_forcings = (
        data_utils.extract_inputs_targets_forcings(
            example_batch,
            target_lead_times=slice("6h", f"{1 * 6}h"),
            **dataclasses.asdict(task_config),
        )
    )

    print("Inputs:  ", eval_inputs.dims.mapping)
    print("Targets: ", eval_targets.dims.mapping)
    print("Forcings:", eval_forcings.dims.mapping)

    def with_configs(fn):
        return functools.partial(fn, model_config=model_config, task_config=task_config)

    # Always pass params and state, so the usage below are simpler
    def with_params(fn):
        return functools.partial(fn, params=params, state=state)

    # Our models aren't stateful, so the state is always empty, so just return the
    # predictions. This is requiredy by our rollout code, and generally simpler.
    def drop_state(fn):
        return lambda **kw: fn(**kw)[0]

    init_jitted = jax.jit(with_configs(run_forward.init))

    if params is None:
        params, state = init_jitted(
            rng=jax.random.PRNGKey(0),
            inputs=train_inputs,
            targets_template=train_targets,
            forcings=train_forcings,
        )

    drop_state(with_params(
        jax.jit(with_configs(loss_fn.apply))))
    with_params(jax.jit(with_configs(grads_fn)))
    run_forward_jitted = drop_state(
        with_params(with_configs(run_forward.apply)))

    # Autoregressive rollout (loop in python)
    assert model_config.resolution in (0, 360.0 / eval_inputs.sizes["lon"]), (
        "Model resolution doesn't match the data resolution. You likely want to "
        "re-filter the dataset list, and download the correct data."
    )

    # rollout推理
    predictions = rollout.chunked_prediction(
        run_forward_jitted,
        rng=jax.random.PRNGKey(0),
        inputs=eval_inputs,
        targets_template=eval_targets * np.nan,
        forcings=eval_forcings,
    )

    # 提取中国区域的数据（经纬度范围：东经73°至135°，北纬18°至54°）
    lat_min, lat_max = 18, 54
    lon_min, lon_max = 73, 135

    # 提取中国区域数据
    china_predictions = predictions.sel(
        lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)
    )
    china_eval = eval_targets.sel(
        lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)
    )
    # 推理结果可视化
    save_var_diff(china_eval, china_predictions,
                  FLAGS.var, FLAGS.level)
    print(
        "----------------------------graphcast inference results----------------------------"
    )
    print(predictions)

    # # train
    # loss, diagnostics = loss_fn_jitted(
    # rng=jax.random.PRNGKey(0),
    # inputs=train_inputs,
    # targets=train_targets,
    # forcings=train_forcings)
    # print("Loss:", float(loss))

    # # Gradient computation (backprop through time)
    # loss, diagnostics, next_state, grads = grads_fn_jitted(
    # inputs=train_inputs,
    # targets=train_targets,
    # forcings=train_forcings)
    # mean_grad = np.mean(jax.tree_util.tree_flatten(jax.tree_util.tree_map(lambda x: np.abs(x).mean(), grads))[0])
    # print(f"Loss: {loss:.4f}, Mean |grad|: {mean_grad:.6f}")

    # predictions = run_forward_jitted(
    # rng=jax.random.PRNGKey(0),
    # inputs=train_inputs,
    # targets_template=train_targets * np.nan,
    # forcings=train_forcings)
    # print(predictions)


if __name__ == "__main__":
    main()
