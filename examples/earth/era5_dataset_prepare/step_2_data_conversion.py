import glob
import os

import h5py
import numpy as np
import xarray as xr


NC_ROOT = "./nc"
TMP_H5_ROOT = "./tmp_h5"
YEARS = list(range(1979, 2026))

SINGLE_LEVEL_VARIABLES = [
    "total_precipitation",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
    "surface_pressure",
    "total_column_water_vapour",
    "sea_surface_temperature",
]
PRESSURE_VARIABLES = [
    "geopotential",
    "relative_humidity",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
]

SINGLE_LEVEL_NAME_MAP = {
    "tp": "total_precipitation",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "t2m": "2m_temperature",
    "msl": "mean_sea_level_pressure",
    "sp": "surface_pressure",
    "tcwv": "total_column_water_vapour",
    "sst": "sea_surface_temperature",
}
PRESSURE_NAME_MAP = {
    "z": "geopotential",
    "r": "relative_humidity",
    "q": "specific_humidity",
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "w": "vertical_velocity",
}


def _as_time_lat_lon(values, source):
    data = np.asarray(values)
    while data.ndim > 3 and 1 in data.shape:
        data = np.squeeze(data)
    if data.ndim != 3:
        raise ValueError(f"{source} must be [T, H, W], got {data.shape}")
    return data.astype(np.float32, copy=False)


def _save_variable(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("fields", data=data)
    print(f"Saved {path}")


def process_single_level_file(nc_path, output_dir):
    ds = xr.open_dataset(nc_path)
    for short_name in ds.data_vars:
        var_name = SINGLE_LEVEL_NAME_MAP.get(short_name, short_name)
        if var_name is None or var_name not in SINGLE_LEVEL_VARIABLES:
            continue
        data = _as_time_lat_lon(ds[short_name].values, nc_path)
        _save_variable(os.path.join(output_dir, f"{var_name}.h5"), data)


def process_pressure_file(nc_path, output_dir):
    ds = xr.open_dataset(nc_path)
    pressure_dim = "pressure_level"
    if pressure_dim not in ds.dims:
        raise ValueError(f"{nc_path} does not contain '{pressure_dim}' dimension")

    for short_name in ds.data_vars:
        var_name = PRESSURE_NAME_MAP.get(short_name, short_name)
        if var_name is None or var_name not in PRESSURE_VARIABLES:
            continue

        for level in ds[pressure_dim].values:
            data = _as_time_lat_lon(
                ds[short_name].sel({pressure_dim: level}).values,
                nc_path,
            )
            out_name = f"{var_name}_{int(level)}.h5"
            _save_variable(os.path.join(output_dir, out_name), data)


def main():
    for year in YEARS:
        year_nc_dir = os.path.join(NC_ROOT, str(year))
        year_tmp_dir = os.path.join(TMP_H5_ROOT, str(year))

        for var_name in SINGLE_LEVEL_VARIABLES:
            nc_path = os.path.join(year_nc_dir, f"{var_name}.nc")
            if os.path.exists(nc_path):
                process_single_level_file(nc_path, year_tmp_dir)

        for var_name in PRESSURE_VARIABLES:
            for nc_path in sorted(glob.glob(os.path.join(year_nc_dir, f"{var_name}_pre*.nc"))):
                process_pressure_file(nc_path, year_tmp_dir)


if __name__ == "__main__":
    main()
