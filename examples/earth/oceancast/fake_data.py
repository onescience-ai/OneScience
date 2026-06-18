from pathlib import Path

import h5py
import numpy as np

from onescience.utils.YParams import YParams


ROOT_DIR = Path(__file__).resolve().parent
CFG_DATA = YParams(str(ROOT_DIR / "conf" / "config.yaml"), "datapipe")
DATASET = CFG_DATA.dataset

DATASET_DIMS = {"T": 20, "H": 160, "W": 360}


def base_data_type(data_type):
    return data_type[:-9] if data_type.endswith("_Forecast") else data_type


WAVE_TYPES = sorted(
    {
        *[base_data_type(item) for item in DATASET.output_types if item.startswith("Wave_")],
        *[base_data_type(item) for item in DATASET.input_types if item.startswith("Wave_")],
    }
)
WIND_RAW_TYPES = ["Wind_U10", "Wind_V10"]
WIND_PROCESSED_TYPES = sorted(
    {
        "Wind_Sin",
        "Wind_Cos",
        "Wind_Strength",
        *[base_data_type(item) for item in DATASET.input_types if item.startswith("Wind_")],
        *[base_data_type(item) for item in DATASET.output_types if item.startswith("Wind_")],
    }
)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def create_sparse_year_file(root_dir, var_name, year, dims, fill_value):
    ensure_dir(root_dir / var_name)
    file_path = root_dir / var_name / f"{year}.h5"
    with h5py.File(file_path, "w") as h5_file:
        h5_file.create_dataset(
            "fields",
            shape=(dims["T"], dims["H"], dims["W"]),
            dtype="float32",
            chunks=(1, dims["H"], dims["W"]),
            fillvalue=fill_value,
        )
    return file_path


def generate_fake_h5():
    years = DATASET.train_time + DATASET.val_time + DATASET.test_time
    ocean_root = ROOT_DIR / DATASET.ocean_data_dir
    wind_uv_root = ROOT_DIR / DATASET.wind_uv_dir
    wind_data_root = ROOT_DIR / DATASET.wind_data_dir

    for year in years:
        for var_name in WAVE_TYPES:
            create_sparse_year_file(ocean_root, var_name, year, DATASET_DIMS, 1.0)
        for var_name in WIND_RAW_TYPES:
            create_sparse_year_file(wind_uv_root, var_name, year, DATASET_DIMS, 1.0)
        for var_name in WIND_PROCESSED_TYPES:
            create_sparse_year_file(wind_data_root, var_name, year, DATASET_DIMS, 1.0)
        print(f"  generated fake year: {year}")


def generate_stats():
    stats_dir = ROOT_DIR / DATASET.stats_dir
    ensure_dir(stats_dir)
    stats_vars = sorted(
        {
            *[
                base_data_type(item)
                for item in DATASET.input_types
                if not base_data_type(item).endswith("Sin") and not base_data_type(item).endswith("Cos")
            ],
            *[base_data_type(item) for item in DATASET.output_types],
        }
    )
    for var_name in stats_vars:
        np.save(stats_dir / f"{var_name}_means.npy", np.zeros(DATASET.img_size, dtype=np.float32))
        np.save(stats_dir / f"{var_name}_stds.npy", np.ones(DATASET.img_size, dtype=np.float32))
    print(f"  stats saved -> {stats_dir}")


def generate_mask():
    static_dir = ROOT_DIR / DATASET.static_dir
    ensure_dir(static_dir)
    mask = np.ones(DATASET.img_size, dtype=np.float32)
    np.save(static_dir / "ocean_mask.npy", mask)
    print(f"  mask saved -> {static_dir / 'ocean_mask.npy'}")


if __name__ == "__main__":
    generate_fake_h5()
    generate_stats()
    generate_mask()
    print("\n✅ Fake datasets generated.")
