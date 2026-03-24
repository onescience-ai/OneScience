"""
统一的JSON转换器

支持在不同格式之间转换JSON数据
"""

import copy
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from biotite.structure import AtomArray, get_chain_starts

from onescience.datapipes.biology.common.json.json_parser import JSONData, JSONParser
from onescience.datapipes.biology.common.json.json_writer import ProteinJSONWriter

logger = logging.getLogger(__name__)


class JSONConverter:
    """
    统一的JSON转换器
    
    支持功能：
    - JSON格式验证和修复
    - 不同版本格式之间的转换
    - 合并多个JSON
    - 拆分JSON
    """
    
    @staticmethod
    def normalize(json_data: Union[Dict[str, Any], JSONData]) -> Dict[str, Any]:
        """
        规范化JSON数据格式
        
        Parameters
        ----------
        json_data : Union[Dict[str, Any], JSONData]
            输入数据
            
        Returns
        -------
        Dict[str, Any]
            规范化后的数据
        """
        if isinstance(json_data, JSONData):
            data = copy.deepcopy(json_data.data)
        else:
            data = copy.deepcopy(json_data)
        
        # 确保有name字段
        if "name" not in data:
            data["name"] = ""
        
        # 确保有sequences字段
        if "sequences" not in data:
            data["sequences"] = []
        
        # 确保covalent_bonds是列表
        if "covalent_bonds" in data and data["covalent_bonds"] is None:
            data["covalent_bonds"] = []
        
        return data
    
    @staticmethod
    def merge(json_data_list: List[Union[Dict[str, Any], JSONData]],
             merge_name: str = "merged") -> Dict[str, Any]:
        """
        合并多个JSON数据
        
        Parameters
        ----------
        json_data_list : List[Union[Dict[str, Any], JSONData]]
            JSON数据列表
        merge_name : str
            合并后的名称
            
        Returns
        -------
        Dict[str, Any]
            合并后的数据
        """
        all_sequences = []
        all_bonds = []
        
        for json_data in json_data_list:
            if isinstance(json_data, JSONData):
                data = json_data.data
            else:
                data = json_data
            
            # 合并序列
            if "sequences" in data:
                all_sequences.extend(data["sequences"])
            
            # 合并共价键
            if "covalent_bonds" in data:
                all_bonds.extend(data["covalent_bonds"])
        
        merged = {
            "name": merge_name,
            "sequences": all_sequences
        }
        
        if all_bonds:
            merged["covalent_bonds"] = all_bonds
        
        return merged
    
    @staticmethod
    def split_by_entity(json_data: Union[Dict[str, Any], JSONData]) -> List[Dict[str, Any]]:
        """
        按实体拆分JSON数据
        
        Parameters
        ----------
        json_data : Union[Dict[str, Any], JSONData]
            输入数据
            
        Returns
        -------
        List[Dict[str, Any]]
            拆分后的数据列表，每个包含一个实体
        """
        if isinstance(json_data, JSONData):
            base_name = json_data.name or "entity"
            data = json_data.data
        else:
            base_name = json_data.get("name", "entity")
            data = json_data
        
        sequences = data.get("sequences", [])
        result = []
        
        for i, entity_dict in enumerate(sequences):
            for entity_type, entity_info in entity_dict.items():
                name = f"{base_name}_{entity_type}_{i+1}"
                split_data = {
                    "name": name,
                    "sequences": [{entity_type: entity_info}]
                }
                result.append(split_data)
        
        return result
    
    @staticmethod
    def to_protenix_format(json_data: Union[Dict[str, Any], JSONData]) -> Dict[str, Any]:
        """
        转换为Protenix格式
        
        Parameters
        ----------
        json_data : Union[Dict[str, Any], JSONData]
            输入数据
            
        Returns
        -------
        Dict[str, Any]
            Protenix格式的数据
        """
        data = JSONConverter.normalize(json_data)
        
        # Protenix格式要求sequences中的每个实体都有count字段
        for entity_dict in data.get("sequences", []):
            for entity_type, entity_info in entity_dict.items():
                if "count" not in entity_info:
                    entity_info["count"] = 1
        
        return data
    
    @staticmethod
    def from_protenix_format(json_data: Union[Dict[str, Any], JSONData]) -> Dict[str, Any]:
        """
        从Protenix格式转换（当前已是Protenix格式，此函数用于未来扩展）
        
        Parameters
        ----------
        json_data : Union[Dict[str, Any], JSONData]
            输入数据
            
        Returns
        -------
        Dict[str, Any]
            转换后的数据
        """
        return JSONConverter.normalize(json_data)


class ProteinJSONConverter(JSONConverter):
    """
    蛋白质结构预测专用的JSON转换器
    
    针对Protenix/AlphaFold3等模型的输入格式进行优化
    """
    
    @staticmethod
    def merge_covalent_bonds(covalent_bonds: List[Dict[str, Any]],
                            all_entity_counts: Dict[str, int]) -> List[Dict[str, Any]]:
        """
        合并具有相同实体和位置的共价键
        
        参考Protenix中的实现
        
        Parameters
        ----------
        covalent_bonds : List[Dict[str, Any]]
            共价键列表
        all_entity_counts : Dict[str, int]
            每个实体的链数量
            
        Returns
        -------
        List[Dict[str, Any]]
            合并后的共价键列表
        """
        bonds_recorder = defaultdict(list)
        bonds_entity_counts = {}
        
        for bond_dict in covalent_bonds:
            bond_unique_string = []
            entity_counts = (
                all_entity_counts.get(str(bond_dict.get("entity1", "")), 0),
                all_entity_counts.get(str(bond_dict.get("entity2", "")), 0)
            )
            
            for i in range(2):
                for key in ["entity", "position", "atom"]:
                    k = f"{key}{i+1}"
                    if k in bond_dict:
                        bond_unique_string.append(str(bond_dict[k]))
            
            bond_unique_string = "_".join(bond_unique_string)
            bonds_recorder[bond_unique_string].append(bond_dict)
            bonds_entity_counts[bond_unique_string] = entity_counts
        
        merged_covalent_bonds = []
        for key, bonds in bonds_recorder.items():
            counts1, counts2 = bonds_entity_counts[key]
            
            if counts1 == counts2 == len(bonds) and len(bonds) > 0:
                # 可以合并
                bond_dict_copy = copy.deepcopy(bonds[0])
                # 移除copy字段
                bond_dict_copy.pop("copy1", None)
                bond_dict_copy.pop("copy2", None)
                merged_covalent_bonds.append(bond_dict_copy)
            else:
                merged_covalent_bonds.extend(bonds)
        
        return merged_covalent_bonds
    
    @staticmethod
    def extract_sequences(json_data: Union[Dict[str, Any], JSONData]) -> Dict[str, List[str]]:
        """
        提取所有序列信息
        
        Parameters
        ----------
        json_data : Union[Dict[str, Any], JSONData]
            输入数据
            
        Returns
        -------
        Dict[str, List[str]]
            按类型分组的序列字典
        """
        if isinstance(json_data, JSONData):
            data = json_data.data
        else:
            data = json_data
        
        sequences_by_type = {
            "proteinChain": [],
            "dnaSequence": [],
            "rnaSequence": [],
            "ligand": [],
            "ion": []
        }
        
        for entity_dict in data.get("sequences", []):
            for seq_type, entity_info in entity_dict.items():
                if seq_type in sequences_by_type:
                    if "sequence" in entity_info:
                        sequences_by_type[seq_type].append(entity_info["sequence"])
                    elif "ligand" in entity_info:
                        sequences_by_type[seq_type].append(entity_info["ligand"])
                    elif "ion" in entity_info:
                        sequences_by_type[seq_type].append(entity_info["ion"])
        
        return sequences_by_type
    
    @staticmethod
    def calculate_composition(json_data: Union[Dict[str, Any], JSONData]) -> Dict[str, int]:
        """
        计算结构组成
        
        Parameters
        ----------
        json_data : Union[Dict[str, Any], JSONData]
            输入数据
            
        Returns
        -------
        Dict[str, int]
            组成统计信息
        """
        if isinstance(json_data, JSONData):
            data = json_data.data
        else:
            data = json_data
        
        composition = {
            "num_entities": 0,
            "num_chains": 0,
            "num_residues": 0,
            "num_protein_chains": 0,
            "num_dna_chains": 0,
            "num_rna_chains": 0,
            "num_ligands": 0,
            "num_ions": 0
        }
        
        for entity_dict in data.get("sequences", []):
            for seq_type, entity_info in entity_dict.items():
                composition["num_entities"] += 1
                count = entity_info.get("count", 1)
                composition["num_chains"] += count
                
                if "sequence" in entity_info:
                    seq_len = len(entity_info["sequence"])
                    composition["num_residues"] += seq_len * count
                
                if seq_type == "proteinChain":
                    composition["num_protein_chains"] += count
                elif seq_type == "dnaSequence":
                    composition["num_dna_chains"] += count
                elif seq_type == "rnaSequence":
                    composition["num_rna_chains"] += count
                elif seq_type == "ligand":
                    composition["num_ligands"] += count
                elif seq_type == "ion":
                    composition["num_ions"] += count
        
        return composition
    
    @staticmethod
    def add_entity_ids(json_data: Union[Dict[str, Any], JSONData]) -> Dict[str, Any]:
        """
        为每个实体添加ID
        
        Parameters
        ----------
        json_data : Union[Dict[str, Any], JSONData]
            输入数据
            
        Returns
        -------
        Dict[str, Any]
            添加了实体ID的数据
        """
        if isinstance(json_data, JSONData):
            data = copy.deepcopy(json_data.data)
        else:
            data = copy.deepcopy(json_data)
        
        entity_id = 1
        for entity_dict in data.get("sequences", []):
            for entity_info in entity_dict.values():
                entity_info["entity_id"] = entity_id
                entity_id += 1
        
        return data
    
    @staticmethod
    def convert_modifications_format(json_data: Union[Dict[str, Any], JSONData],
                                    target_format: str = "protenix") -> Dict[str, Any]:
        """
        转换修饰格式
        
        Parameters
        ----------
        json_data : Union[Dict[str, Any], JSONData]
            输入数据
        target_format : str
            目标格式（当前仅支持"protenix"）
            
        Returns
        -------
        Dict[str, Any]
            转换后的数据
        """
        if isinstance(json_data, JSONData):
            data = copy.deepcopy(json_data.data)
        else:
            data = copy.deepcopy(json_data)
        
        for entity_dict in data.get("sequences", []):
            for seq_type, entity_info in entity_dict.items():
                if "modifications" in entity_info:
                    mods = entity_info["modifications"]
                    converted_mods = []
                    
                    for mod in mods:
                        if isinstance(mod, list) and len(mod) == 2:
                            # 旧格式: [position, "CCD_XXX"]
                            position, mod_type = mod
                            if seq_type == "proteinChain":
                                converted_mods.append({
                                    "ptmPosition": position,
                                    "ptmType": mod_type
                                })
                            else:
                                converted_mods.append({
                                    "basePosition": position,
                                    "modificationType": mod_type
                                })
                        elif isinstance(mod, dict):
                            # 新格式，保持原样
                            converted_mods.append(mod)
                    
                    entity_info["modifications"] = converted_mods
        
        return data
    
    @staticmethod
    def create_bond_dict(entity1: int, position1: int, atom1: str,
                        entity2: int, position2: int, atom2: str,
                        copy1: Optional[int] = None,
                        copy2: Optional[int] = None) -> Dict[str, Any]:
        """
        创建共价键字典
        
        Parameters
        ----------
        entity1, entity2 : int
            实体ID
        position1, position2 : int
            位置
        atom1, atom2 : str
            原子名称
        copy1, copy2 : Optional[int]
            副本ID
            
        Returns
        -------
        Dict[str, Any]
            共价键字典
        """
        bond = {
            "entity1": entity1,
            "position1": position1,
            "atom1": atom1,
            "entity2": entity2,
            "position2": position2,
            "atom2": atom2
        }
        
        if copy1 is not None:
            bond["copy1"] = copy1
        if copy2 is not None:
            bond["copy2"] = copy2
        
        return bond
    
    @staticmethod
    def filter_by_entity_type(json_data: Union[Dict[str, Any], JSONData],
                             include_types: Optional[List[str]] = None,
                             exclude_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        按实体类型过滤
        
        Parameters
        ----------
        json_data : Union[Dict[str, Any], JSONData]
            输入数据
        include_types : Optional[List[str]]
            包含的类型列表（None表示包含所有）
        exclude_types : Optional[List[str]]
            排除的类型列表
            
        Returns
        -------
        Dict[str, Any]
            过滤后的数据
        """
        if isinstance(json_data, JSONData):
            data = copy.deepcopy(json_data.data)
            name = json_data.name
        else:
            data = copy.deepcopy(json_data)
            name = data.get("name", "")
        
        include_types = include_types or ["proteinChain", "dnaSequence", "rnaSequence", "ligand", "ion"]
        exclude_types = exclude_types or []
        
        filtered_sequences = []
        for entity_dict in data.get("sequences", []):
            for seq_type in list(entity_dict.keys()):
                if seq_type in include_types and seq_type not in exclude_types:
                    filtered_sequences.append({seq_type: entity_dict[seq_type]})
        
        result = {
            "name": name,
            "sequences": filtered_sequences
        }
        
        # 保留其他字段
        for key in ["covalent_bonds", "assembly_id"]:
            if key in data:
                result[key] = data[key]
        
        return result


class JSONBatchProcessor:
    """
    JSON批量处理器
    
    用于批量处理多个JSON文件
    """
    
    def __init__(self, 
                 parser: Optional[JSONParser] = None,
                 converter: Optional[JSONConverter] = None,
                 writer: Optional[ProteinJSONWriter] = None):
        """
        Parameters
        ----------
        parser : Optional[JSONParser]
            JSON解析器
        converter : Optional[JSONConverter]
            JSON转换器
        writer : Optional[ProteinJSONWriter]
            JSON写入器
        """
        self.parser = parser or JSONParser()
        self.converter = converter or JSONConverter()
        self.writer = writer or ProteinJSONWriter()
    
    def batch_convert(self,
                     input_paths: List[Union[str, Path]],
                     output_dir: Union[str, Path],
                     operation: str = "normalize") -> List[Path]:
        """
        批量转换JSON文件
        
        Parameters
        ----------
        input_paths : List[Union[str, Path]]
            输入文件路径列表
        output_dir : Union[str, Path]
            输出目录
        operation : str
            操作类型："normalize", "split", "merge"
            
        Returns
        -------
        List[Path]
            输出文件路径列表
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_paths = []
        
        if operation == "merge":
            # 合并所有文件
            json_data_list = []
            names = []
            
            for path in input_paths:
                try:
                    json_data = self.parser.parse_file(path)
                    json_data_list.append(json_data)
                    names.append(json_data.name or Path(path).stem)
                except Exception as e:
                    logger.warning(f"Failed to parse {path}: {e}")
                    continue
            
            if json_data_list:
                merged = self.converter.merge(json_data_list, merge_name="merged_batch")
                output_path = output_dir / "merged.json"
                self.writer.write(merged, output_path, name="merged_batch")
                output_paths.append(output_path)
        
        elif operation == "split":
            # 拆分每个文件
            for path in input_paths:
                try:
                    json_data = self.parser.parse_file(path)
                    split_data_list = self.converter.split_by_entity(json_data)
                    
                    for i, split_data in enumerate(split_data_list):
                        output_path = output_dir / f"{Path(path).stem}_entity_{i+1}.json"
                        self.writer.write(split_data, output_path, name=split_data.get("name", ""))
                        output_paths.append(output_path)
                
                except Exception as e:
                    logger.warning(f"Failed to process {path}: {e}")
                    continue
        
        else:  # normalize
            # 规范化每个文件
            for path in input_paths:
                try:
                    json_data = self.parser.parse_file(path)
                    normalized = self.converter.normalize(json_data)
                    
                    output_path = output_dir / f"{Path(path).stem}_normalized.json"
                    self.writer.write(normalized, output_path, name=normalized.get("name", ""))
                    output_paths.append(output_path)
                
                except Exception as e:
                    logger.warning(f"Failed to process {path}: {e}")
                    continue
        
        return output_paths
