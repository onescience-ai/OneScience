import logging
from typing import Union, Dict, Any, List, Optional
import ase.io
from torch.utils.data import Dataset

from ..base import MaterialsDataset
from ..core.atomic_data import AtomicData
from ..core.utils import Configuration, config_from_atoms, KeySpecification
from ...tools.keys import DefaultKeys

# ✨ 新增：创建一个混合配置类，既支持 ["key"] 访问，也支持 .key 访问
class HybridConfig(dict):
    def __getattr__(self, attr):
        # 当尝试访问 .verbose 时，如果没有该属性，返回 False (默认值)
        return self.get(attr, False)
    
    def __setattr__(self, key, value):
        self[key] = value

class TextDataset(MaterialsDataset):
    """
    基于文本的原子构型数据集 (L3)。
    """
    
    def __init__(self, 
                 file_path: Optional[Union[str, List[str]]] = None,
                 configurations: Optional[List[Configuration]] = None,
                 r_max: float = 4.0,                    
                 z_table: Any = None,                    
                 head: str = "Default",             
                 heads: Optional[List[str]] = None,
                 **kwargs):

        # ✨ 关键修复：构造一个满足 BaseDataset 要求的 config 对象
        # 只要让它具备 .verbose 属性即可，同时保留它作为字典的特性以防万一
        dummy_config = HybridConfig()
        dummy_config.verbose = kwargs.get("verbose", False) # 设置 verbose，默认为 False
        
        # 1. 调用 L2 基类，传入 dummy_config 而不是 None
        super().__init__(
            config=dummy_config,
            r_max=r_max,
            z_table=z_table
        )
        
        self.mode = kwargs.get("mode", "train")
        self.kwargs = kwargs
        self.kwargs["head"] = head
        self.kwargs["heads"] = heads
        self.transform = kwargs.get("transform", None)
        
        # (由于 BaseDataset 初始化日志时用到了 config.verbose，现在这步应该可以通过了)
        self.logger.info(f"TextDataset (L3) initialized with r_max={self.r_max}")

        # 2. 加载数据逻辑
        if configurations is not None:
            self.logger.info(f"Initializing dataset from {len(configurations)} in-memory configurations.")
            self.configurations = configurations
        elif file_path is not None:
            self.logger.info(f"Loading ALL atoms from text files into memory: {file_path}")
            self.configurations = self._load_from_text(file_path)
            self.logger.info(f" > Loaded {len(self.configurations)} samples from file.")
        else:
            raise ValueError("TextDataset requires either 'file_path' or 'configurations' to be provided.")

    def _load_from_text(self, file_path: Union[str, List[str]]) -> List[Configuration]:
        if isinstance(file_path, str):
            file_path = [file_path]
            
        all_configs = []
        key_specification = KeySpecification.from_defaults()
        
        config_type_weights = self.kwargs.get("config_type_weights", {"Default": 1.0})
        head_name = self.kwargs.get("head", "Default")

        for path in file_path:
            self.logger.debug(f"Reading from file: {path}")
            try:
                atoms_list = ase.io.read(path, index=":")
                if not isinstance(atoms_list, list):
                    atoms_list = [atoms_list]
                    
                for atoms in atoms_list:
                    atoms.info[DefaultKeys.HEAD_KEY.value] = head_name
                    
                    config = config_from_atoms(
                        atoms,
                        key_specification=key_specification,
                        config_type_weights=config_type_weights,
                        head_name=head_name,
                    )
                    all_configs.append(config)
                    
            except Exception as e:
                self.logger.error(f"Failed to read atoms from text file {path}: {e}")
                
        return all_configs

    def __len__(self):
        return len(self.configurations)

    def __getitem__(self, index):
        # ... (这部分代码保持不变) ...
        config = self.configurations[index]
        try:
            atomic_data = AtomicData.from_config(
                config,
                z_table=self.z_table,
                cutoff=self.r_max,
                heads=self.kwargs.get("heads", ["Default"]),
            )
        except Exception as e:
            self.logger.error(f"Error creating AtomicData at index {index}: {e}")
            return None

        if self.transform:
            atomic_data = self.transform(atomic_data)
        
        return atomic_data