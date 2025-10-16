import os
import h5py
import json
import numpy as np
import xarray as xr
from onescience.utils.fcn.YParams import YParams


def generate_debug_sample(cfg, years=None, timestamp="2001010100"):
    years = years or list(range(2001, 2011))
    total_files = 1460
    variables = cfg.channels
    M = len(variables)
    shape = (M, 721, 1440)
    real_data = np.random.randn(*shape).astype(np.float32)

    data_root = os.path.join(cfg.data_dir, "data")
    os.makedirs(data_root, exist_ok=True)
    real_year = years[0]
    real_dir = os.path.join(data_root, str(real_year))
    os.makedirs(real_dir, exist_ok=True)

    real_path = os.path.join(real_dir, f"{timestamp}.h5")
    with h5py.File(real_path, "w") as f:
        f.create_dataset("fields", data=real_data)
    print(f"✅ Real data file: {real_path}")

    # 当前年份其余时间步软链
    for i in range(total_files):
        ts = (np.datetime64(f"{real_year}-01-01T00") + np.timedelta64(i * 6, "h")).astype(str).replace("-", "").replace("T", "")[:10]
        path = os.path.join(real_dir, f"{ts}.h5")
        if path != real_path and not os.path.exists(path):
            os.symlink(f"{timestamp}.h5", path)

    # 其他年份软链所有时间步
    for y in years[1:]:
        y_dir = os.path.join(data_root, str(y))
        os.makedirs(y_dir, exist_ok=True)
        for i in range(total_files):
            ts = (np.datetime64(f"{y}-01-01T00") + np.timedelta64(i * 6, "h")).astype(str).replace("-", "").replace("T", "")[:10]
            path = os.path.join(y_dir, f"{ts}.h5")
            if not os.path.exists(path):
                rel = os.path.relpath(real_path, y_dir)
                os.symlink(rel, path)

    print("✅ Soft links for all years generated.")

    # 生成 metadata.json
    meta = {
        "years": [str(y) for y in years],
        "variables": variables,
        "total_files": total_files
    }
    meta_path = os.path.join(cfg.data_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✅ Metadata written to {meta_path}")


def get_stats(cfg):
    arr = np.random.randn(1, len(cfg.channels), 1, 1).astype(np.float32) # 保存数据
    os.makedirs(cfg.stats_dir, exist_ok=True)
    np.save(f'{cfg.stats_dir}/global_stds.npy', arr) 
    np.save(f'{cfg.stats_dir}/global_means.npy', arr)
    print(f"✅ Stats data: {arr.shape}, dtype: {arr.dtype}, save to {cfg.stats_dir}")


def get_static(var, name):
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
    arr = np.random.randn(721, 1440).astype(np.float32) # 保存数据
    np.save(f'{cfg.static_dir}/land_mask.npy', arr) 
    np.save(f'{cfg.static_dir}/soil_type.npy', arr) 
    np.save(f'{cfg.static_dir}/topography.npy', arr)
    print(f"✅ Static data: {arr.shape}, dtype: {arr.dtype}, save to {cfg.static_dir}")

# === Example Usage ===
if __name__ == "__main__":
    cfg = YParams('conf/config.yaml', 'model')
    generate_debug_sample(cfg)
    get_stats(cfg)
    get_static('z', 'geopotential')
    get_static('lsm', 'land_sea_mask')