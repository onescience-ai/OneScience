import xarray as xr
import h5py
import os
import numpy as np
import sys


pressure_var_map = {
    'z': 'geopotential',
    'r': 'relative_humidity',
    'q': 'specific_humidity',
    't': 'temperature',
    'u': 'u_component_of_wind',
    'v': 'v_component_of_wind'
}
land_var_map = {
    'u10': '10m_u_component_of_wind',
    'v10': '10m_v_component_of_wind',
    't2m': '2m_temperature',
    'msl': 'mean_sea_level_pressure',
    'sp': 'surface_pressure',
    'tcwv': 'total_column_water_vapour',
    'tp': 'total_precipitation'
}


def process_pressure_file(nc_path: str, output_dir: str = "."):
    
    ds = xr.open_dataset(nc_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for var_name in ds.data_vars:
        var = ds[var_name]
        pressure_dim = 'pressure_level'
        levels = ds[pressure_dim].values
        for i, level in enumerate(levels):
            sliced = var.sel({pressure_dim: level})
            filename = f"{pressure_var_map[var_name]}_{int(level)}.h5"
            save_path = os.path.join(output_dir, filename)
            if os.path.exists(save_path):
                print(f'{save_path} exists, skipping...')
            else:
                save_variable_to_h5(var_name, sliced.values, save_path)


def save_variable_to_h5(var_name: str, data: np.ndarray, h5_filename: str):
    with h5py.File(h5_filename, 'w') as h5f:
        h5f.create_dataset('fields', data=data)
    print(f"✅ Saved {h5_filename}")


def process_land_file(nc_path: str, output_dir: str = "."):
    ds = xr.open_dataset(nc_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for var_name in ds.data_vars:
        print(f'processing {var_name}...')
        var = ds[var_name]
        filename = f"{land_var_map[var_name]}.h5"
        save_path = os.path.join(output_dir, filename)
        if os.path.exists(save_path):
            print(f'{save_path} exists, skipping...')
        else:
            print(save_path)
            save_variable_to_h5(var_name, var.values, save_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: input the filename")
        sys.exit(1)
    
    filename = sys.argv[1]
     
    base_path = "/public/onestore/onedatasets/ERA5/newh5/newdata/"
    for year in range(2001, 2021):
        output_folder = f"{base_path}/tmp_h5/{year}" 
        nc_file = f"{base_path}/nc/tmpnc/{year}/{filename}.nc"
        if filename[-4:-1] == 'pre':
            process_pressure_file(nc_file, output_folder)
        else:
            process_land_file(nc_file, output_folder)
## 输入文件名，不用带nc