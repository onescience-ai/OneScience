"""
This code handles the downloading, downsampling, and conversion of ocean physics reanalysis data from the Copernicus Marine Data Store.

1. Data Sources:
   - Wave Data: GLOBAL_MULTIYEAR_WAV_001_032 (https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_WAV_001_032/description)
       - wave height
       - wave period
       - wave direction
   - Sea Surface Data: GLOBAL_MULTIYEAR_PHY_001_030 (https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description)
       - sea surface height 
       - sea surface salinity 
       - sea surface temperature 
   - Wind Data: WIND_GLO_PHY_L4_MY_012_006 (https://data.marine.copernicus.eu/product/WIND_GLO_PHY_L4_MY_012_006/description)
       - wind_u
       - wind_v
   - The download code maybe failed due to connection error. 
   - If you encounter the connection error, please try several times. 
   - If it still doesn't work,please go to the website and use mannual download.

2. Functionalities:
   - Data Download: Fetches wave and wind reanalysis data.
   - Data Downsampling: Reduces the spatial resolution of data.
   - Data Conversion: Converts `.nc` files to `.h5` format for model training.

3. Usage:
   - Run the script using:
       python copernicusmarine_wave_mirror.py start_year, start_month, start_day, end_year, end_month, end_day
     Example for one year: 
       python copernicusmarine_wave_mirror.py 2015 1 1 2015 12 31
     Example for a single day:
       python copernicusmarine_wave_mirror.py 2015 8 1 2015 8 1

4. Default Download Settings:
   - Spatial Coverage: Global area ([-180, 180], [-90, 90])
   - Temporal Coverage: Wave data (daily), Wind data (3-hour intervals)

5. Important Notes:
   - Ensure that each download covers at most one year and does not span across two years (Because the train code split data according to year index).
   - `wave.nc` contains: wave_period, wave_height, wave_direction.
   - `wind.nc` contains: eastward_wind, northward_wind.
   - Special handling is required for wind data before and after 2009. Refer to the code for details.

6. Prerequisite:
   - Install the `copernicusmarine` package (using conda install).
   - Create an account at https://data.marine.copernicus.eu/products and configure your credentials in the `download` function.
"""



import sys
import datetime
import calendar
import copernicusmarine
import os
import h5py
import xarray as xr
import numpy as np

# Replacing username and password to your account.
username="*"
password="*"

def download_wave(start_year, start_month, start_day, end_year, end_month, end_day, out_dir):
    print(f"Downloading wave...", end='\t')
    copernicusmarine.subset(
        dataset_version="202411",
        minimum_longitude=-180,
        maximum_longitude=179.8000030517578,
        minimum_latitude=-89.80000305175781,
        maximum_latitude=89.80000305175781,
        force_download=True,
        subset_method="strict",
        disable_progress_bar=True,

        dataset_id='cmems_mod_glo_wav_my_0.2deg_PT3H-i', 
        variables=['VHM0', 'VTM02', 'VMDR'],
        start_datetime=f"{start_year}-{start_month}-{start_day}T00:00:00",
        end_datetime= f"{end_year}-{end_month}-{end_day}T00:00:00", 

        username=username,   
        password=password,
        output_directory=f"{out_dir}/",
        output_filename=f"wave_{start_year}.nc"
    )
    print(f"successfully")


def download_phy(start_year, start_month, start_day, end_year, end_month, end_day, out_dir):
    print(f"Downloading phy...", end='\t')
    copernicusmarine.subset(
        dataset_version="202311",
        minimum_longitude=-180,
        maximum_longitude=179.9166717529297,
        minimum_latitude=-80,
        maximum_latitude=90,
        minimum_depth=0.49402499198913574,
        maximum_depth=0.49402499198913574,
        force_download=True,
        subset_method="strict",
        disable_progress_bar=True,

        dataset_id='cmems_mod_glo_phy_my_0.083deg_P1D-m',
        variables=['thetao', 'so', 'zos'],
        start_datetime=f"{start_year}-{start_month}-{start_day}T00:00:00",
        end_datetime= f"{end_year}-{end_month}-{end_day}T00:00:00", 

        username=username,   
        password=password,
        output_directory=f"{out_dir}/",
        output_filename=f"phy_{start_year}.nc",
    )
    print(f"successfully")

    
def download_wind(start_year, start_month, start_day, end_year, end_month, end_day, out_dir):
    #If you want to download data before 2009, replace following key-values in code
    #dataset_id="cmems_obs-wind_glo_phy_my_l4_0.25deg_PT1H",
    #dataset_version="202406",
    #minimum_latitude=-89.875, 
    #maximum_latitude=89.875

    #If you want to download data after 2009, replace following key-values in code
    #dataset_id='cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H',
    #dataset_version="202211",
    #minimum_latitude=-89.9375
    #maximum_latitude=89.9375
    print(f"Downloading wind...", end='\t')
    copernicusmarine.subset(
        dataset_id="cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H",
        dataset_version="202211",
        minimum_latitude=-89.9375,
        maximum_latitude=89.9375,
        minimum_longitude=-179.9375,
        maximum_longitude=179.9375,
        force_download=True,
        subset_method="strict",
        disable_progress_bar=True,

        variables=['eastward_wind','northward_wind'],
        start_datetime=f"{start_year}-{start_month}-{start_day}T00:00:00",
        end_datetime= f"{end_year}-{end_month}-{end_day}T00:00:00", 
        username=username,   
        password=password,
        output_directory=f"{out_dir}/",
        output_filename=f"wind_{start_year}.nc"
    )
    print(f"successfully")
    


def merge(ori_file_path, tar_file_path):
    ds_high = xr.open_dataset(ori_file_path)
    lat_new = np.arange(-90, 90, 1.0)
    lon_new = np.arange(-180, 179.5, 1.0)
    ds_low_res = ds_high.interp(latitude=lat_new, longitude=lon_new, method="linear")
    ds_low_res.to_netcdf(tar_file_path)
    print(tar_file_path, ' merge done...')

    
def convert_wave_nc_to_h5(inp_dir, out_dir, year):
    try:
        wave_file = f'{inp_dir}/wave_{year}_new.nc'
        wave_var_to_filename = {
            'VHM0': f'Wave_Height/{year}.h5',
            'VTM02': f'Wave_Period/{year}.h5',
            'VMDR': f'Wave_Direction/{year}.h5'
        }
        wave_ds = xr.open_dataset(wave_file)
        for var_name, file_name in wave_var_to_filename.items():
            output_file = os.path.join(out_dir, file_name)
            if os.path.exists(output_file):
                print(f"{output_file} already exists. Skipping conversion.")
                continue
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            var_data = wave_ds[var_name].values
            with h5py.File(output_file, 'w') as h5_data:
                h5_data.create_dataset(var_name, data=var_data)
            print(f"Successfully converted {var_name} from {wave_file} to {output_file}")
        wave_ds.close()

    except Exception as e:
        print(f"Error converting {wave_file}: {e}")


def convert_phy_nc_to_h5(inp_dir, out_dir, year):
    try:
        phy_file = f'{inp_dir}/phy_{year}.nc'
        phy_var_to_filename = {
            'thetao': f'Ocean_SST/{year}.h5',
            'so': f'Ocean_SSS/{year}.h5',
            'zos': f'Ocean_SSH/{year}.h5'
        }
        phy_ds = xr.open_dataset(phy_file)
        for var_name, file_name in phy_var_to_filename.items():
            output_file = os.path.join(out_dir, file_name)
            if os.path.exists(output_file):
                print(f"{output_file} already exists. Skipping conversion.")
                continue
            os.makedirs(os.path.dirname(output_file), exist_ok=True)  # 创建文件夹
            var_data = phy_ds[var_name].values
            var_data = var_data.squeeze()
            with h5py.File(output_file, 'w') as h5_data:
                h5_data.create_dataset(var_name, data=var_data)
            print(f"Successfully converted {var_name} from {phy_file} to {output_file}")
        phy_ds.close()

    except Exception as e:
        print(f"Error converting {phy_file}: {e}")



def convert_wind_nc_to_h5(inp_dir, out_dir, year):
    try:
        wind_file = f'{inp_dir}/wind_{year}_new.nc'
        wind_var_to_filename = {
            'eastward_wind': f'Wind_U10/{year}.h5',
            'northward_wind': f'Wind_V10/{year}.h5',
        }
        wind_ds = xr.open_dataset(wind_file)
        for var_name, file_name in wind_var_to_filename.items():
            output_file = os.path.join(out_dir, file_name)
            if os.path.exists(output_file):
                print(f"{output_file} already exists. Skipping conversion.")
                continue
            os.makedirs(os.path.dirname(output_file), exist_ok=True) 
            var_data = wind_ds[var_name].values
            var_data = var_data[::3] #3 hours
            with h5py.File(output_file, 'w') as h5_data:
                h5_data.create_dataset(var_name, data=var_data)
            print(f"Successfully converted {var_name} from {wind_file} to {output_file}")
        wind_ds.close()
    except Exception as e:
        print(f"Error converting {wind_file}: {e}")
        

if __name__ == '__main__':
    if len(sys.argv) != 7:
        print("Usage: python copernicusmarine_wave_mirror.py  <start_year> <start_month> <start_day> <end_year> <end_month> <end_day>")
        sys.exit(1)
    
    start_year = int(sys.argv[1])
    start_month = int(sys.argv[2])
    start_day = int(sys.argv[3])
    end_year = int(sys.argv[4])
    end_month = int(sys.argv[5])
    end_day = int(sys.argv[6])
    out_dir=f"oceandata/{start_year}"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f'downloading data from {start_year}/{start_month}/{start_day} to {end_year}/{end_month}/{end_day}, data will be saved at {out_dir}')
    download_wave(start_year, start_month, start_day, end_year, end_month, end_day, out_dir)
    download_wind(start_year, start_month, start_day, end_year, end_month, end_day, out_dir)
    download_phy(start_year, start_month, start_day, end_year, end_month, end_day, out_dir)
    
    print(f'down sampling data')
    merge(f'{out_dir}/wave_{start_year}.nc', f'{out_dir}/wave_{start_year}_new.nc')
    merge(f'{out_dir}/phy_{start_year}.nc', f'{out_dir}/wind_{start_year}_new.nc')
    merge(f'{out_dir}/wind_{start_year}.nc', f'{out_dir}/wind_{start_year}_new.nc')
    
    print(f'convert data to HDF5')
    convert_wave_nc_to_h5(out_dir, out_dir, start_year)
    convert_phy_nc_to_h5(out_dir, out_dir, start_year)
    convert_wind_nc_to_h5(out_dir, out_dir, start_year)
