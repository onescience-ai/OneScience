import gzip
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Union

import pandas as pd

import onescience.datapipes.protenix
import onescience.models.protenix
from onescience.utils.protenix.torch_utils import map_values_to_list

# 创建一个虚拟模块，指向新的模块路径
sys.modules["protenix.data"] = onescience.datapipes.protenix
sys.modules["protenix"] = onescience.models.protenix

PANDAS_NA_VALUES = [
    "",
    "#N/A",
    "#N/A N/A",
    "#NA",
    "-1.#IND",
    "-1.#QNAN",
    "-NaN",
    "-nan",
    "1.#IND",
    "1.#QNAN",
    "<NA>",
    "N/A",
    # "NA",
    "NULL",
    "NaN",
    "n/a",
    "nan",
    "null",
]


def read_indices_csv(csv: Union[str, Path]) -> pd.DataFrame:
    """
    Read a csv file without the content changing.

    Args:
        csv (Union[str, Path]): A csv file path.

    Returns:
        pd.DataFrame : A pandas DataFrame.
    """
    df = pd.read_csv(csv, na_values=PANDAS_NA_VALUES, keep_default_na=False, dtype=str)
    return df


def load_gzip_pickle(pkl: Union[str, Path]) -> Any:
    """
    Load a gzip pickle file.

    Args:
        pkl (Union[str, Path]): A gzip pickle file path.

    Returns:
        Any: The loaded data.
    """
    with gzip.open(pkl, "rb") as f:
        data = pickle.load(f)
    return data


def dump_gzip_pickle(data: Any, pkl: Union[str, Path]):
    """
    Dump a gzip pickle file.

    Args:
        data (Any): The data to be dumped.
        pkl (Union[str, Path]): A gzip pickle file path.
    """
    with gzip.open(pkl, "wb") as f:
        pickle.dump(data, f)


def save_json(data: dict, output_fpath: Union[str, Path], indent: int = 4):
    """
    Save a dictionary to a JSON file.

    Args:
        data (dict): The dictionary to be saved.
        output_fpath (Union[str, Path]): The output file path.
        indent (int, optional): The indentation level for the JSON file. Defaults to 4.
    """
    data_json = data.copy()
    data_json = map_values_to_list(data_json)
    with open(output_fpath, "w") as f:
        if indent is not None:
            json.dump(data_json, f, indent=indent)
        else:
            json.dump(data_json, f)
