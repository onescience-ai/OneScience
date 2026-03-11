# ERA5 Data Prepare

Here contains some steps to construct era5 dataset for onescience-earth models.

## Download the data from ECMWF

Firstly, you need to sign up and get the cdsapi-key from official website.
https://cds.climate.copernicus.eu/datasets

Mostly used two dataset: 

land: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview

pressure: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=overview

1. Select the variables, year, month, day, time, pressure_level to get the example download code from bottom of website.
2. Copy and change the corresponding code to "step_1_data_download.py",  focus on the following fields: product_type, variable, year, month, day, pressure_level, time.
3. Confirm the download file path and file name, default named as "{path}/{year}/var_pre{i}.nc" or "{path}/{year}/var.nc", for example, './2000/temperature_pre1.nc' or './2000/2m_temperature.nc'. 
4. Notice that, limited by the file size, ECMWF mostly download 5 pressure levels of one variable per year (6 hours).
5. From your profile get the cdsapi key: https://cds.climate.copernicus.eu/profile. Using "vim ~/.cdsapirc" and add the url+key from the profile website. Format like:

```
url: https://cds.climate.copernicus.eu/api
key: xx-xx-xx-xx-xx
```

6. running the "step_1_data_download.py" to get the data. Notice that, this file contains both land and pressure variables, you can select the variables to download.

## Data conversion from NC to H5

In this step, the downloaded *.nc file will be converted to *.h5

1. make directories and some files and make sure the ERA5 data path is constructed as following:

```Properties
data_path
│   metadata.json (must contains, create it manually for the first time)
└───stats/
└───static/
└───nc/ (downloaded files)
└───h5/ (temporary folder for this step, you can delete after data construction)
└───tmp_h5/ (temporary folder for this step, you can delete after data construction)
└─────────────────────

the metadata.json contains the info as follows:：
{
    "years": [
        "1979",
        "1980",
        ...
        "2025"
    ],
    "variables": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        ...
        "vertical_velocity_500"
    ],
    "total_files": 1460
}

 - "years" is the time range you downloaded

 - "variables" contains all land and pressure variables, notice that, the land-variables are directly named, while the pressure-variables are named as "var_pressure", such as "vertical_velocity_500" represent vertical_velocity in 500 hPa;

 - "total_files" represent the number of files per year, which also represents the time-resolution(total_files//365). Here we drop the last day of leap year, for example, the last time of 2000 is 2000123018 (6 hours resolution).

```

2. run the "step_2_data_conversion.py", fix the "pressure_var_list" and "land_var_list" to your downloaded dataset variables. The  temporary data will be saved to './tmp_h5/'

## Data merge

In this step, the data orignized by variables are merge to data orignized by datetime.

Running "step_3_data_merge.py".

After this step, the './h5/' contains the whole dataset used by onescience, you should rename this folder to 'data'.

## Stats calculate

In this step, the means and stds files are calculated, which are used to normalized the model input data. 

This process may cost some time, please ensure the server is always online.

running "step_4_stats_calculate.py"

## Final state

After the above processes, the dataset will be constructed, and the final data directory are as follows:

```Properties
data_path
│   metadata.json
└───stats
│   │   global_stds.npy
│   │   global_means.npy
└───static/ (add files if you need, such as land_sea_mask, topography, etc.)
└───data
└──────1979
│   │   1979010100.h5
│   │   ...
│   │   1979123118.h5
└──────1980
│   │   1980010100.h5
│   │   ...
│   │   1980123118.h5
└──────2025
│   │   2025010100.h5
│   │   ...
│   │   2025123118.h5
└─────────────────────


 - the shapes of 'data_path/stats/global_means.npy && global_stds.npy' are all [1, C, 1, 1], where C is the number of variables in metadata.json.

 - the file name format in 'data_path/data/year/' follows 'yyyymmddhh.h5', where 'yyyy' represents year, 'mm' represents month, dd represents day, hh represents hour.

 - each h5 file data field named 'fields', contains [C, H, W] data, where H and W are height and width of images, respectively. Default H and W are 721 and 1440.
 
 - the order of C is strictly following the the order of variables in metadata.
```