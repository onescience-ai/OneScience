import os

import h5py
import numpy as np


TMP_H5_ROOT = "./tmp_h5"
OUTPUT_ROOT = "./data"
YEARS = list(range(1979, 2026))
TIME_STEP_HOURS = 6
WRITE_CHUNK_STEPS = 8

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
PRESSURE_LEVELS = [
    1, 2, 3, 5, 7,
    10, 20, 30, 70, 125,
    175, 225, 350, 450, 550,
    650, 750, 775, 800, 825,
    875, 900, 950, 975,
]

VARIABLES = SINGLE_LEVEL_VARIABLES + [
    f"{var}_{level}"
    for var in PRESSURE_VARIABLES
    for level in PRESSURE_LEVELS
]


def _variable_file_map(year_dir, variables):
    file_map = {}
    missing = []
    for var_name in variables:
        path = os.path.join(year_dir, f"{var_name}.h5")
        if os.path.exists(path):
            file_map[var_name] = path
        else:
            missing.append(var_name)
    if missing:
        raise FileNotFoundError(f"Missing variables under {year_dir}: {missing}")
    return file_map


def _read_variable_shape(path):
    with h5py.File(path, "r") as f:
        return f["fields"].shape


def merge_year(year):
    year_dir = os.path.join(TMP_H5_ROOT, str(year))
    file_map = _variable_file_map(year_dir, VARIABLES)
    first_shape = _read_variable_shape(file_map[VARIABLES[0]])
    time_steps, height, width = first_shape

    output_path = os.path.join(OUTPUT_ROOT, f"{year}.h5")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    with h5py.File(output_path, "w") as out_f:
        fields = out_f.create_dataset(
            "fields",
            shape=(time_steps, len(VARIABLES), height, width),
            dtype="float32",
            chunks=(1, len(VARIABLES), height, width),
        )
        fields.attrs["variables"] = VARIABLES
        fields.attrs["time_step"] = TIME_STEP_HOURS

        for channel_idx, var_name in enumerate(VARIABLES):
            with h5py.File(file_map[var_name], "r") as in_f:
                var_data = in_f["fields"]
                if var_data.shape != first_shape:
                    raise ValueError(
                        f"{file_map[var_name]} shape {var_data.shape} "
                        f"does not match {first_shape}"
                    )
                for start in range(0, time_steps, WRITE_CHUNK_STEPS):
                    end = min(start + WRITE_CHUNK_STEPS, time_steps)
                    fields[start:end, channel_idx, :, :] = np.asarray(
                        var_data[start:end],
                        dtype=np.float32,
                    )

    print(f"Saved {output_path}: ({time_steps}, {len(VARIABLES)}, {height}, {width})")


def main():
    for year in YEARS:
        merge_year(year)


if __name__ == "__main__":
    main()
