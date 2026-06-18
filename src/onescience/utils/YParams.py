# from ruamel.yaml import YAML
# import logging
# import re
# import os


# def parse_env_vars(data):
#     """递归解析YAML数据中的环境变量"""
#     if isinstance(data, str):
#         # 匹配 ${ENV_VAR} 或 $ENV_VAR 格式
#         env_var_pattern = r"\$\{(\w+)\}|\$(\w+)"
#         return re.sub(
#             env_var_pattern,
#             lambda m: os.environ.get(m.group(1) or m.group(2), ""),
#             data,
#         )
#     elif isinstance(data, dict):
#         return {k: parse_env_vars(v) for k, v in data.items()}
#     elif isinstance(data, list):
#         return [parse_env_vars(item) for item in data]
#     return data


# class YParams:
#     """Yaml file parser"""

#     def __init__(self, yaml_filename, config_name, print_params=False):
#         self._yaml_filename = yaml_filename
#         self._config_name = config_name
#         self.params = {}

#         if print_params:
#             print("------------------ Configuration ------------------")

#         with open(yaml_filename) as _file:
#             data = YAML().load(_file)
#             # 解析环境变量
#             data = parse_env_vars(data)

#             for key, val in data[config_name].items():
#                 if print_params:
#                     print(key, val)
#                 if val == "None":
#                     val = None

#                 self.params[key] = val
#                 self.__setattr__(key, val)

#         if print_params:
#             print("---------------------------------------------------")

#     def __getitem__(self, key):
#         return self.params[key]

#     def __setitem__(self, key, val):
#         self.params[key] = val
#         self.__setattr__(key, val)

#     def __contains__(self, key):
#         return key in self.params

#     def update_params(self, config):
#         for key, val in config.items():
#             self.params[key] = val
#             self.__setattr__(key, val)

#     def log(self):
#         logging.info("------------------ Configuration ------------------")
#         logging.info("Configuration file: " + str(self._yaml_filename))
#         logging.info("Configuration name: " + str(self._config_name))
#         for key, val in self.params.items():
#             logging.info(str(key) + " " + str(val))
#         logging.info("---------------------------------------------------")

from ruamel.yaml import YAML
import logging
import re
import os


def parse_env_vars(data):
    """递归解析YAML数据中的环境变量"""
    if isinstance(data, str):
        env_var_pattern = r"\$\{(\w+)\}|\$(\w+)"
        return re.sub(
            env_var_pattern,
            lambda m: os.environ.get(m.group(1) or m.group(2), ""),
            data,
        )
    elif isinstance(data, dict):
        return {k: parse_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [parse_env_vars(item) for item in data]
    return data


class AttrDict(dict):
    """递归支持点号访问的字典"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 初始化时递归转换
        for k, v in self.items():
            self[k] = self._wrap(v)

    def __setitem__(self, key, value):
        super().__setitem__(key, self._wrap(value))

    def _wrap(self, value):
        if isinstance(value, dict):
            return AttrDict(value)
        elif isinstance(value, list):
            return [self._wrap(v) for v in value]
        return value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"No such attribute: {key}")

    def __setattr__(self, key, value):
        self[key] = self._wrap(value)

    def __delattr__(self, key):
        del self[key]

    def __repr__(self):
        return f"AttrDict({dict.__repr__(self)})"
    
    # --- [ 新增的修复方法 ] ---
    def to_dict(self):
        """
        递归地将 AttrDict 转换回标准的 dict。
        """
        plain_dict = {}
        for k, v in self.items():
            if isinstance(v, AttrDict):
                plain_dict[k] = v.to_dict()
            elif isinstance(v, list):
                plain_dict[k] = self._convert_list(v)
            else:
                plain_dict[k] = v
        return plain_dict

    def _convert_list(self, value_list):
        """ 辅助 to_dict，用于处理列表中的 AttrDicts """
        new_list = []
        for item in value_list:
            if isinstance(item, AttrDict):
                new_list.append(item.to_dict())
            elif isinstance(item, list):
                new_list.append(self._convert_list(item))
            else:
                new_list.append(item)
        return new_list


class YParams:
    """Yaml file parser（增强版，支持递归 params.xxx.yyy 调用）"""

    def __init__(self, yaml_filename, config_name, print_params=False):
        self._yaml_filename = yaml_filename
        self._config_name = config_name

        if print_params:
            print("------------------ Configuration ------------------")

        with open(yaml_filename) as _file:
            data = YAML().load(_file)
            data = parse_env_vars(data)

            # ✅ 递归转 AttrDict
            config_data = AttrDict(data[config_name])
            self.params = config_data

            # ✅ 同步顶层属性，方便直接访问
            for key, val in config_data.items():
                object.__setattr__(self, key, val)
                if print_params:
                    print(f"{key}: {val}")

        if print_params:
            print("---------------------------------------------------")

    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key, val):
        self.params[key] = val
        object.__setattr__(self, key, self.params[key])

    def __contains__(self, key):
        return key in self.params

    def __iter__(self):
        return iter(self.params)

    def __len__(self):
        return len(self.params)

    def get(self, key, default=None):
        return self.params.get(key, default)

    def keys(self):
        return self.params.keys()

    def items(self):
        return self.params.items()

    def values(self):
        return self.params.values()

    def update_params(self, config):
        for key, val in config.items():
            self[key] = val
            
    def to_dict(self):
        return self.params.to_dict()
    
    def log(self):
        logging.info("------------------ Configuration ------------------")
        logging.info(f"Configuration file: {self._yaml_filename}")
        logging.info(f"Configuration name: {self._config_name}")
        for key, val in self.params.items():
            logging.info(f"{key}: {val}")
        logging.info("---------------------------------------------------")
