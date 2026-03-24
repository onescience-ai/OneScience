"""
统一的JSON解析器

支持从各种文件格式解析JSON数据
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from onescience.datapipes.biology.common.utils.file_utils import open_file

logger = logging.getLogger(__name__)


@dataclass
class JSONData:
    """
    统一的JSON数据格式
    
    Attributes
    ----------
    data : Dict[str, Any]
        JSON数据字典
    name : str
        数据名称（如样本名称）
    source_path : Optional[Path]
        源文件路径
    metadata : Dict[str, Any]
        额外的元数据
    """
    data: Dict[str, Any]
    name: str = ""
    source_path: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """验证数据一致性"""
        if not isinstance(self.data, dict):
            raise ValueError(
                f"JSON data must be a dictionary. Got {type(self.data)}"
            )
    
    def __getitem__(self, key: str) -> Any:
        """通过键访问数据"""
        return self.data[key]
    
    def __contains__(self, key: str) -> bool:
        """检查是否包含指定键"""
        return key in self.data
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取指定键的值，如果不存在则返回默认值"""
        return self.data.get(key, default)
    
    def get_sequences(self) -> List[Dict[str, Any]]:
        """
        获取sequences字段
        
        Returns
        -------
        List[Dict[str, Any]]
            序列列表
        """
        return self.data.get("sequences", [])
    
    def get_covalent_bonds(self) -> List[Dict[str, Any]]:
        """
        获取covalent_bonds字段
        
        Returns
        -------
        List[Dict[str, Any]]
            共价键列表
        """
        return self.data.get("covalent_bonds", [])
    
    def has_sequence_type(self, seq_type: str) -> bool:
        """
        检查是否包含指定类型的序列
        
        Parameters
        ----------
        seq_type : str
            序列类型，如"proteinChain", "dnaSequence", "rnaSequence", "ligand", "ion"
            
        Returns
        -------
        bool
            是否包含该类型
        """
        for entity_dict in self.get_sequences():
            if seq_type in entity_dict:
                return True
        return False
    
    def get_sequence_count(self) -> int:
        """
        获取序列数量
        
        Returns
        -------
        int
            序列数量
        """
        return len(self.get_sequences())


class JSONParser:
    """
    统一的JSON解析器
    
    支持功能：
    - 从文件解析JSON
    - 从字符串解析JSON
    - 验证JSON格式
    - 批量解析多个文件
    """
    
    @staticmethod
    def parse_string(json_string: str) -> JSONData:
        """
        从字符串解析JSON
        
        Parameters
        ----------
        json_string : str
            JSON格式的字符串
            
        Returns
        -------
        JSONData
            解析后的JSON数据对象
        """
        try:
            data = json.loads(json_string)
            
            # 如果解析结果是列表，取第一个元素（兼容Protenix格式）
            if isinstance(data, list) and len(data) > 0:
                data = data[0]
            
            # 提取名称
            name = data.get("name", "")
            
            return JSONData(
                data=data,
                name=name
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {e}")
    
    @staticmethod
    def parse_file(path: Union[str, Path], encoding: str = "utf-8") -> JSONData:
        """
        从文件解析JSON（支持压缩文件）
        
        Parameters
        ----------
        path : Union[str, Path]
            JSON文件路径（支持.gz, .bz2, .xz压缩文件）
        encoding : str
            文件编码，默认为utf-8
            
        Returns
        -------
        JSONData
            解析后的JSON数据对象
        """
        path = Path(path)
        
        try:
            with open_file(path, 'r', encoding=encoding) as f:
                content = f.read()
            
            json_data = JSONParser.parse_string(content)
            json_data.source_path = path
            json_data.metadata["source_format"] = path.suffix.lstrip(".")
            
            return json_data
        except FileNotFoundError:
            raise FileNotFoundError(f"JSON file not found: {path}")
        except Exception as e:
            raise ValueError(f"Failed to parse JSON file {path}: {e}")
    
    @staticmethod
    def parse_files(paths: List[Union[str, Path]], 
                   encoding: str = "utf-8") -> List[JSONData]:
        """
        批量解析多个JSON文件
        
        Parameters
        ----------
        paths : List[Union[str, Path]]
            JSON文件路径列表
        encoding : str
            文件编码，默认为utf-8
            
        Returns
        -------
        List[JSONData]
            解析后的JSON数据对象列表
        """
        results = []
        for path in paths:
            try:
                json_data = JSONParser.parse_file(path, encoding)
                results.append(json_data)
            except Exception as e:
                logger.warning(f"Failed to parse {path}: {e}")
                continue
        return results
    
    @staticmethod
    def validate(json_data: Union[str, Dict[str, Any]], 
                 required_fields: Optional[List[str]] = None) -> bool:
        """
        验证JSON数据格式
        
        Parameters
        ----------
        json_data : Union[str, Dict[str, Any]]
            JSON字符串或字典
        required_fields : Optional[List[str]]
            必需字段列表
            
        Returns
        -------
        bool
            验证是否通过
        """
        try:
            if isinstance(json_data, str):
                data = json.loads(json_data)
                if isinstance(data, list) and len(data) > 0:
                    data = data[0]
            else:
                data = json_data
            
            if not isinstance(data, dict):
                return False
            
            if required_fields:
                for field in required_fields:
                    if field not in data:
                        logger.warning(f"Required field '{field}' not found in JSON")
                        return False
            
            return True
        except (json.JSONDecodeError, TypeError):
            return False
    
    @staticmethod
    def extract_entities(json_data: JSONData) -> List[Dict[str, Any]]:
        """
        从JSON数据中提取所有实体信息
        
        Parameters
        ----------
        json_data : JSONData
            JSON数据对象
            
        Returns
        -------
        List[Dict[str, Any]]
            实体信息列表，每个实体包含type和info
        """
        entities = []
        sequences = json_data.get_sequences()
        
        for entity_dict in sequences:
            for entity_type, entity_info in entity_dict.items():
                entities.append({
                    "type": entity_type,
                    "info": entity_info,
                    "count": entity_info.get("count", 1)
                })
        
        return entities
    
    @staticmethod
    def get_entity_types(json_data: JSONData) -> List[str]:
        """
        获取JSON数据中所有实体类型
        
        Parameters
        ----------
        json_data : JSONData
            JSON数据对象
            
        Returns
        -------
        List[str]
            实体类型列表
        """
        types = []
        sequences = json_data.get_sequences()
        
        for entity_dict in sequences:
            types.extend(entity_dict.keys())
        
        return types


class ProteinJSONParser(JSONParser):
    """
    蛋白质结构预测专用的JSON解析器
    
    针对Protenix/AlphaFold3等模型的输入格式进行优化
    """
    
    REQUIRED_FIELDS = ["sequences"]
    
    @staticmethod
    def validate_protein_json(json_data: Union[str, Dict[str, Any]]) -> bool:
        """
        验证是否为有效的蛋白质结构预测JSON
        
        Parameters
        ----------
        json_data : Union[str, Dict[str, Any]]
            JSON字符串或字典
            
        Returns
        -------
        bool
            验证是否通过
        """
        return JSONParser.validate(
            json_data, 
            required_fields=ProteinJSONParser.REQUIRED_FIELDS
        )
    
    @staticmethod
    def get_sequence_info(json_data: JSONData) -> Dict[str, List[Dict[str, Any]]]:
        """
        获取序列详细信息
        
        Parameters
        ----------
        json_data : JSONData
            JSON数据对象
            
        Returns
        -------
        Dict[str, List[Dict[str, Any]]]
            按类型分组的序列信息
        """
        info = {
            "proteinChain": [],
            "dnaSequence": [],
            "rnaSequence": [],
            "ligand": [],
            "ion": []
        }
        
        sequences = json_data.get_sequences()
        entity_id = 1
        
        for entity_dict in sequences:
            for seq_type, seq_info in entity_dict.items():
                if seq_type in info:
                    info[seq_type].append({
                        "entity_id": entity_id,
                        "info": seq_info
                    })
                entity_id += 1
        
        return info
    
    @staticmethod
    def count_atoms_estimate(json_data: JSONData) -> int:
        """
        估算原子数量（粗略估计）
        
        Parameters
        ----------
        json_data : JSONData
            JSON数据对象
            
        Returns
        -------
        int
            估算的原子数量
        """
        total_atoms = 0
        sequences = json_data.get_sequences()
        
        # 平均每个残基的原子数
        ATOMS_PER_RESIDUE = {
            "proteinChain": 7,
            "dnaSequence": 20,
            "rnaSequence": 20,
            "ligand": 20,
            "ion": 1
        }
        
        for entity_dict in sequences:
            for seq_type, seq_info in entity_dict.items():
                count = seq_info.get("count", 1)
                atoms_per_unit = ATOMS_PER_RESIDUE.get(seq_type, 10)
                
                if "sequence" in seq_info:
                    seq_len = len(seq_info["sequence"])
                elif "ligand" in seq_info:
                    # 配体通常是一个分子
                    seq_len = 1
                elif "ion" in seq_info:
                    seq_len = 1
                else:
                    seq_len = 1
                
                total_atoms += count * seq_len * atoms_per_unit
        
        return total_atoms
