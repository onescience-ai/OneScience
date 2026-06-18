import os
import h5py
import numpy as np
import xarray as xr
from onescience.utils.YParams import YParams


# 各数据集固定的空间和时间维度
DATASET_DIMS = {"T": 10, "H": 721, "W": 1440, "time_step": 6}


def generate_fake_h5(data_dir, var_names, years, dims):
    """
    为每个年份生成一个空 h5 文件。
    利用 HDF5 chunked 压缩数据集未写入 chunk 即返回 fill_value=0 的特性，
    文件实际只含元数据，极小，但 shape 与真实数据完全一致。
    """
    os.makedirs(os.path.join(data_dir, "data"), exist_ok=True)
    T, C = dims["T"], len(var_names)
    H, W = dims["H"], dims["W"]

    for year in years:
        path = os.path.join(data_dir, "data", f"{year}.h5")
        with h5py.File(path, "w") as f:
            ds = f.create_dataset(
                "fields",
                shape=(T, C, H, W),
                dtype="float32",
                chunks=(1, C, H, W),
                fillvalue=0.0,
            )
            ds.attrs["variables"] = var_names
            ds.attrs["time_step"] = dims["time_step"]

        size_kb = os.path.getsize(path) / 1024
        print(f"  {year}.h5  shape=({T},{C},{H},{W})  "
              f"logical={T*C*H*W*4/1024**3:.1f}GB  actual={size_kb:.1f}KB")


def generate_fake_npy(result_dir, n_vars, years, dims):
    """
    为 base/short/medium/long 各阶段生成假的模型输出 npy 文件。
    只创建一个真实的 npy 文件，其余使用 symlink。
    """
    T = dims["T"]
    H, W = dims["H"], dims["W"]
    time_step = dims["time_step"]

    data_root = os.path.join(result_dir, "data")
    os.makedirs(data_root, exist_ok=True)

    real_year = years[0]
    real_dir = os.path.join(data_root, str(real_year))
    os.makedirs(real_dir, exist_ok=True)

    timestamp = f'{real_year}010100'
    real_path = os.path.join(real_dir, f"{timestamp}.npy")
    real_data = np.zeros((n_vars, H, W), dtype=np.float32)
    np.save(real_path, real_data)

    # 当前年份其余时间步 symlink
    for i in range(T):
        ts = (np.datetime64(f"{real_year}-01-01T00") + np.timedelta64(i * time_step, "h")
              ).astype(str).replace("-", "").replace("T", "")[:10]
        path = os.path.join(real_dir, f"{ts}.npy")
        if path != real_path and not os.path.exists(path):
            os.symlink(f"{timestamp}.npy", path)

    # 其他年份 symlink
    for y in years[1:]:
        y_dir = os.path.join(data_root, str(y))
        os.makedirs(y_dir, exist_ok=True)
        for i in range(T):
            ts = (np.datetime64(f"{y}-01-01T00") + np.timedelta64(i * time_step, "h")
                  ).astype(str).replace("-", "").replace("T", "")[:10]
            path = os.path.join(y_dir, f"{ts}.npy")
            if not os.path.exists(path):
                rel = os.path.relpath(real_path, y_dir)
                os.symlink(rel, path)

    print(f"  ✅ Fake npy data generated → {result_dir}")


def generate_stats(data_dir, n_vars):
    """生成全零均值、全一标准差的 stats 文件，避免归一化除零。"""
    os.makedirs(data_dir, exist_ok=True)
    shape = (1, n_vars, 1, 1)
    np.save(os.path.join(data_dir, "global_means.npy"),
            np.zeros(shape, dtype=np.float32))
    np.save(os.path.join(data_dir, "global_stds.npy"),
            np.ones(shape,  dtype=np.float32))
    print(f"  stats saved → {data_dir}")


def get_static(cfg, var, name):
    os.makedirs(cfg.static_dir, exist_ok=True)
    ds = xr.Dataset(
        data_vars={
            f"{var}": (("valid_time", "latitude", "longitude"),
                np.random.rand(1, 721, 1440).astype(np.float32))
        },
        coords={
            "valid_time": ["2015-12-31"],
            "latitude": np.linspace(90, -90, 721, dtype=np.float64),
            "longitude": np.linspace(0, 359.75, 1440, dtype=np.float64),
            "number": 0,
            "expver": "",
        },
        attrs={
            "GRIB_centre": "ecmf",
            "GRIB_centreDescription": "European Centre for Medium-Range Weather Forecasts",
            "GRIB_subCentre": "0",
            "Conventions": "CF-1.7",
            "institution": "European Centre for Medium-Range Weather Forecasts",
            "history": "Generated manually",
        }
    )

    ds.to_netcdf(f"{cfg.static_dir}/{name}.nc")
    arr = np.random.randn(721, 1440).astype(np.float32)
    np.save(f'{cfg.static_dir}/land_mask.npy', arr)
    np.save(f'{cfg.static_dir}/soil_type.npy', arr)
    np.save(f'{cfg.static_dir}/topography.npy', arr)
    print(f"✅ Static data: {arr.shape}, dtype: {arr.dtype}, save to {cfg.static_dir}")


if __name__ == "__main__":
    cfg_datapipe = YParams("conf/config.yaml", "datapipe")

    if cfg_datapipe.dataset.data_dir.startswith("/public/") or cfg_datapipe.dataset.data_dir.startswith("/work/"):
        print("请检查 config，确保各 *_dir 指向本地测试路径而非生产路径。")
        exit()

    years    = cfg_datapipe.dataset.train_time + cfg_datapipe.dataset.val_time + cfg_datapipe.dataset.test_time
    atm_vars = cfg_datapipe.dataset.channels
    n_vars   = len(atm_vars)

    # 主 ERA5 数据
    generate_fake_h5(cfg_datapipe.dataset.data_dir, atm_vars, years, DATASET_DIMS)
    stats_dir = os.path.join(cfg_datapipe.dataset.data_dir, "stats")
    generate_stats(stats_dir, n_vars)

    # 各阶段中间模型输出（用于 short/medium/long 微调的输入）
    for stage in ['base', 'short', 'medium', 'long']:
        generate_fake_npy(f'./result/{stage}', n_vars, years, DATASET_DIMS)

    static_dir = os.path.join(cfg_datapipe.dataset.data_dir, "static")
    get_static(static_dir, 'z', 'geopotential')
    get_static(static_dir, 'lsm', 'land_sea_mask')

    print("\n✅ Fake datasets generated.")
