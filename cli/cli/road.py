import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ONESCIENCE_DATASETS_DIR = os.environ.get(
    "ONESCIENCE_DATASETS_DIR",
    "/public/share/sugonhpcapp01/onestore/onedatasets",
)

ONESCIENCE_MODELS_DIR = os.environ.get(
    "ONESCIENCE_MODELS_DIR",
    "/public/share/sugonhpcapp01/onestore/onemodels",
)

DATASET_PATHS = {
    "airfoil": "CFD_Benchmark/airfoil",
    "darcy": "CFD_Benchmark/darcy",
    "elasticity": "CFD_Benchmark/elasticity",
    "ns": "CFD_Benchmark/ns",
    "pipe": "CFD_Benchmark/pipe",
    "plasticity": "CFD_Benchmark/plasticity",
    "era5": "ERA5",
    "era5_stats": "ERA5/stats",
    "era5_static": "ERA5/static",
    "cwb": "corrdiff/cwb",
    "graphcast": "graphcast",
    "mace": "mace",
    "evo2": "evo2",
    "protenix": "protenix",
    "openfold": "openfold",
    "matris": "matris",
    "deepcfd": "DeepCFD",
    "beno": "BENO",
    "pdenneval": "PDENNEval",
    "lagrangian_mgn": "Lagrangian_MGN",
    "topology": "GP_for_TO",
}
