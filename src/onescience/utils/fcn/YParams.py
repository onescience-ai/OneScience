import logging
import os
import re

from ruamel.yaml import YAML


def parse_env_vars(data):
    """递归解析YAML数据中的环境变量"""
    if isinstance(data, str):
        # 匹配 ${ENV_VAR} 或 $ENV_VAR 格式
        env_var_pattern = r"\$\{(\w+)\}|\$(\w+)"
        return re.sub(
            env_var_pattern,
            lambda m: os.environ.get(
                m.group(1) or m.group(2), ""),
            data,
        )
    elif isinstance(data, dict):
        return {k: parse_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [parse_env_vars(item) for item in data]
    return data


class YParams:
    """Yaml file parser"""

    def __init__(self, yaml_filename, config_name, print_params=False):
        self._yaml_filename = yaml_filename
        self._config_name = config_name
        self.params = {}

        if print_params:
            print(
                "------------------ Configuration ------------------")

        with open(yaml_filename) as _file:
            data = YAML().load(_file)
            # 解析环境变量
            data = parse_env_vars(data)

            for key, val in data[config_name].items():
                if print_params:
                    print(key, val)
                if val == "None":
                    val = None

                self.params[key] = val
                self.__setattr__(key, val)

        if print_params:
            print(
                "---------------------------------------------------")

    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key, val):
        self.params[key] = val
        self.__setattr__(key, val)

    def __contains__(self, key):
        return key in self.params

    def update_params(self, config):
        for key, val in config.items():
            self.params[key] = val
            self.__setattr__(key, val)

    def log(self):
        logging.info(
            "------------------ Configuration ------------------")
        logging.info("Configuration file: " +
                     str(self._yaml_filename))
        logging.info("Configuration name: " +
                     str(self._config_name))
        for key, val in self.params.items():
            logging.info(str(key) + " " + str(val))
        logging.info(
            "---------------------------------------------------")
