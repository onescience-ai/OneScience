"""Command factory - registers all CLI commands in a dict like pip's commands_dict."""

from .bench import bench
from .remock import remock
from .help_ import help_cmd
from .list import list_group
from .info import info_group
from .log import log
from .status import status
from .config import config_group
from .train import train
from .infer import infer
from .eval import eval


commands_dict = {
    "bench": bench,
    "remock": remock,
    "help": help_cmd,
    "list": list_group,
    "info": info_group,
    "log": log,
    "status": status,
    "config": config_group,
    "train": train,
    "infer": infer,
    "eval": eval,
}
