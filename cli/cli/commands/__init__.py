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
from .compare import compare
from .pipeline import pipeline
from .data import data_group
from .deploy import deploy_group
from .experiment import experiment_group
from .env import env_group


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
    "compare": compare,
    "pipeline": pipeline,
    "data": data_group,
    "deploy": deploy_group,
    "experiment": experiment_group,
    "env": env_group,
}
