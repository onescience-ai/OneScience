# ERA5 Data Prepare

This directory prepares ERA5 data for the current OneScience ERA5 dataloader.

The final dataset layout is:

```text
era5_root/
└── data/
    ├── 1979.h5
    ├── 1980.h5
    └── ...
```

Each yearly HDF5 file is self-contained — it stores both the data and the
normalization statistics:

```text
fields: [T, C, H, W]
fields.attrs["variables"]: channel names
fields.attrs["time_step"]: hours between samples
global_means: [1, C, 1, 1]
global_stds:  [1, C, 1, 1]
```

This is the format read by `src/onescience/datapipes/climate/era5.py`.

## 1. Account And Environment

Register a Copernicus CDS account:

```text
https://cds.climate.copernicus.eu/
```

Create `~/.cdsapirc` from your CDS profile:

```text
url: https://cds.climate.copernicus.eu/api
key: <your-key>
```

Install dependencies:

```bash
pip install cdsapi xarray netCDF4 h5py numpy
```

## 2. Download NetCDF

Edit the top settings in `step_1_data_download.py`:

```python
pressure_list = [...]
pressure_level = [...]
land_list = [...]
save_path = "./nc"
```

Then run:

```bash
python step_1_data_download.py
```

Output:

```text
./nc/{year}/{variable}.nc
./nc/{year}/{pressure_variable}_pre{i}.nc
```

The default download cadence is 6 hours:

```text
00:00, 06:00, 12:00, 18:00
```

## 3. Convert NetCDF To Temporary HDF5

Edit the top settings in `step_2_data_conversion.py`:

```python
NC_ROOT = "./nc"
TMP_H5_ROOT = "./tmp_h5"
YEARS = list(range(1979, 2026))
SINGLE_LEVEL_VARIABLES = [...]
PRESSURE_VARIABLES = [...]
```

Then run:

```bash
python step_2_data_conversion.py
```

Output:

```text
./tmp_h5/{year}/{variable}.h5
```

These temporary files are organized by variable and are only used by the merge step.

## 4. Merge To Yearly HDF5

Edit the top settings in `step_3_data_merge.py`:

```python
TMP_H5_ROOT = "./tmp_h5"
OUTPUT_ROOT = "./data"
YEARS = list(range(1979, 2026))
TIME_STEP_HOURS = 6
VARIABLES = [...]
```

Then run:

```bash
python step_3_data_merge.py
```

Output:

```text
./data/{year}.h5
```

Each file stores all variables for one year:

```text
fields: [T, C, 721, 1440]
```

## 5. Calculate And Embed Statistics

Edit the top settings in `step_4_stats_calculate.py`:

```python
DATA_DIR = "./data"
CHUNK_SIZE = 100
```

Then run:

```bash
python step_4_stats_calculate.py
```

It accumulates the global mean/std across all years (chunked over time to bound
memory) and writes them back into every yearly HDF5 file as the `global_means` /
`global_stds` datasets:

```text
{year}.h5 / global_means: [1, C, 1, 1]
{year}.h5 / global_stds:  [1, C, 1, 1]
```

For very large datasets you can parallelize the per-year accumulation on a
cluster (one job per year computing a partial `sum` / `sum_sq` / `count`, then a
single reduce + embed). That orchestration is left to your scheduler.

## 6. Use In Model Config

Set the model config dataset path to:

```yaml
dataset:
  data_dir: "<era5_root>"
```

The dataloader reads variables from `fields.attrs["variables"]` and the
normalization statistics from the `global_means` / `global_stds` datasets
inside each yearly HDF5 file, so neither a separate `stats/` directory nor a
`metadata.json` is required.
