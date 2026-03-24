"""
统一的JSON写入器

支持将数据写入JSON文件
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from onescience.datapipes.biology.common.utils.file_utils import open_file

logger = logging.getLogger(__name__)


@dataclass
class JSONWriteConfig:
    """
    JSON写入配置
    
    Attributes
    ----------
    indent : Optional[int]
        缩进空格数，None表示不缩进
    ensure_ascii : bool
        是否确保ASCII编码（False允许非ASCII字符）
    sort_keys : bool
        是否按键排序
    separators : Optional[tuple]
        分隔符元组，如(',', ':')
    wrap_in_list : bool
        是否将数据包装在列表中（Protenix格式）
    """
    indent: Optional[int] = 4
    ensure_ascii: bool = False
    sort_keys: bool = False
    separators: Optional[tuple] = None
    wrap_in_list: bool = False


class JSONWriter:
    """
    统一的JSON写入器
    
    支持功能：
    - 将数据写入JSON文件
    - 支持压缩文件输出
    - 格式化输出
    - 批量写入
    """
    
    def __init__(self, config: Optional[JSONWriteConfig] = None):
        """
        Parameters
        ----------
        config : Optional[JSONWriteConfig]
            写入配置，如果为None则使用默认配置
        """
        self.config = config or JSONWriteConfig()
    
    def write(self, 
              data: Dict[str, Any], 
              path: Union[str, Path],
              name: Optional[str] = None) -> Path:
        """
        将数据写入JSON文件
        
        Parameters
        ----------
        data : Dict[str, Any]
            要写入的数据
        path : Union[str, Path]
            输出文件路径
        name : Optional[str]
            数据名称，如果提供则添加到数据中
            
        Returns
        -------
        Path
            写入的文件路径
        """
        path = Path(path)
        
        # 确保目录存在
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # 准备数据
        write_data = dict(data)
        if name is not None:
            write_data["name"] = name
        
        # 如果需要包装在列表中
        if self.config.wrap_in_list:
            write_data = [write_data]
        
        # 写入文件
        dump_kwargs = {
            "ensure_ascii": self.config.ensure_ascii,
            "sort_keys": self.config.sort_keys,
        }
        
        if self.config.indent is not None:
            dump_kwargs["indent"] = self.config.indent
        
        if self.config.separators is not None:
            dump_kwargs["separators"] = self.config.separators
        
        with open_file(path, 'w', encoding='utf-8') as f:
            json.dump(write_data, f, **dump_kwargs)
        
        logger.debug(f"JSON written to {path}")
        return path
    
    def write_string(self, data: Dict[str, Any], name: Optional[str] = None) -> str:
        """
        将数据转换为JSON字符串
        
        Parameters
        ----------
        data : Dict[str, Any]
            要转换的数据
        name : Optional[str]
            数据名称，如果提供则添加到数据中
            
        Returns
        -------
        str
            JSON字符串
        """
        write_data = dict(data)
        if name is not None:
            write_data["name"] = name
        
        if self.config.wrap_in_list:
            write_data = [write_data]
        
        dump_kwargs = {
            "ensure_ascii": self.config.ensure_ascii,
            "sort_keys": self.config.sort_keys,
        }
        
        if self.config.indent is not None:
            dump_kwargs["indent"] = self.config.indent
        
        if self.config.separators is not None:
            dump_kwargs["separators"] = self.config.separators
        
        return json.dumps(write_data, **dump_kwargs)
    
    def write_batch(self,
                   data_list: List[Dict[str, Any]],
                   path: Union[str, Path],
                   names: Optional[List[str]] = None) -> Path:
        """
        批量写入多个JSON数据到单个文件
        
        Parameters
        ----------
        data_list : List[Dict[str, Any]]
            数据列表
        path : Union[str, Path]
            输出文件路径
        names : Optional[List[str]]
            名称列表，与数据列表一一对应
            
        Returns
        -------
        Path
            写入的文件路径
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # 准备数据
        batch_data = []
        for i, data in enumerate(data_list):
            write_data = dict(data)
            if names and i < len(names):
                write_data["name"] = names[i]
            batch_data.append(write_data)
        
        # 写入文件
        dump_kwargs = {
            "ensure_ascii": self.config.ensure_ascii,
            "sort_keys": self.config.sort_keys,
        }
        
        if self.config.indent is not None:
            dump_kwargs["indent"] = self.config.indent
        
        with open_file(path, 'w', encoding='utf-8') as f:
            json.dump(batch_data, f, **dump_kwargs)
        
        logger.debug(f"Batch JSON written to {path} ({len(batch_data)} items)")
        return path


class ProteinJSONWriter(JSONWriter):
    """
    蛋白质结构预测专用的JSON写入器
    
    针对Protenix/AlphaFold3等模型的输入格式进行优化
    """
    
    def __init__(self, indent: int = 4, wrap_in_list: bool = True):
        """
        Parameters
        ----------
        indent : int
            缩进空格数
        wrap_in_list : bool
            是否将数据包装在列表中（Protenix格式要求）
        """
        config = JSONWriteConfig(
            indent=indent,
            ensure_ascii=False,
            sort_keys=False,
            wrap_in_list=wrap_in_list
        )
        super().__init__(config)
    
    def create_sequence_entry(self,
                             seq_type: str,
                             sequence: Optional[str] = None,
                             ligand: Optional[str] = None,
                             ion: Optional[str] = None,
                             count: int = 1,
                             modifications: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        创建序列条目
        
        Parameters
        ----------
        seq_type : str
            序列类型：proteinChain, dnaSequence, rnaSequence, ligand, ion
        sequence : Optional[str]
            序列字符串（用于proteinChain, dnaSequence, rnaSequence）
        ligand : Optional[str]
            配体CCD代码（用于ligand）
        ion : Optional[str]
            离子类型（用于ion）
        count : int
            副本数量
        modifications : Optional[List[Dict[str, Any]]]
            修饰信息列表
            
        Returns
        -------
        Dict[str, Any]
            序列条目字典
        """
        entry = {"count": count}
        
        if seq_type in ["proteinChain", "dnaSequence", "rnaSequence"]:
            if sequence is None:
                raise ValueError(f"Sequence is required for {seq_type}")
            entry["sequence"] = sequence
        elif seq_type == "ligand":
            if ligand is None:
                raise ValueError("Ligand is required for ligand type")
            entry["ligand"] = ligand
        elif seq_type == "ion":
            if ion is None:
                raise ValueError("Ion is required for ion type")
            entry["ion"] = ion
        else:
            raise ValueError(f"Unknown sequence type: {seq_type}")
        
        if modifications:
            entry["modifications"] = modifications
        
        return {seq_type: entry}
    
    def create_protein_entry(self,
                            sequence: str,
                            count: int = 1,
                            modifications: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        创建蛋白质序列条目
        
        Parameters
        ----------
        sequence : str
            蛋白质序列（单字母代码）
        count : int
            副本数量
        modifications : Optional[List[Dict[str, Any]]]
            修饰信息列表，格式为[{"ptmPosition": 1, "ptmType": "CCD_SEP"}, ...]
            
        Returns
        -------
        Dict[str, Any]
            蛋白质序列条目
        """
        return self.create_sequence_entry(
            seq_type="proteinChain",
            sequence=sequence,
            count=count,
            modifications=modifications
        )
    
    def create_dna_entry(self,
                        sequence: str,
                        count: int = 1,
                        modifications: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        创建DNA序列条目
        
        Parameters
        ----------
        sequence : str
            DNA序列（单字母代码）
        count : int
            副本数量
        modifications : Optional[List[Dict[str, Any]]]
            修饰信息列表，格式为[{"basePosition": 1, "modificationType": "CCD_MA6"}, ...]
            
        Returns
        -------
        Dict[str, Any]
            DNA序列条目
        """
        return self.create_sequence_entry(
            seq_type="dnaSequence",
            sequence=sequence,
            count=count,
            modifications=modifications
        )
    
    def create_rna_entry(self,
                        sequence: str,
                        count: int = 1,
                        modifications: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        创建RNA序列条目
        
        Parameters
        ----------
        sequence : str
            RNA序列（单字母代码）
        count : int
            副本数量
        modifications : Optional[List[Dict[str, Any]]]
            修饰信息列表
            
        Returns
        -------
        Dict[str, Any]
            RNA序列条目
        """
        return self.create_sequence_entry(
            seq_type="rnaSequence",
            sequence=sequence,
            count=count,
            modifications=modifications
        )
    
    def create_ligand_entry(self,
                           ligand: str,
                           count: int = 1) -> Dict[str, Any]:
        """
        创建配体条目
        
        Parameters
        ----------
        ligand : str
            配体CCD代码（如"CCD_ATP"）或SMILES字符串
        count : int
            副本数量
            
        Returns
        -------
        Dict[str, Any]
            配体条目
        """
        return self.create_sequence_entry(
            seq_type="ligand",
            ligand=ligand,
            count=count
        )
    
    def create_ion_entry(self,
                        ion: str,
                        count: int = 1) -> Dict[str, Any]:
        """
        创建离子条目
        
        Parameters
        ----------
        ion : str
            离子类型（如"NA", "CL", "MG", "ZN"等）
        count : int
            副本数量
            
        Returns
        -------
        Dict[str, Any]
            离子条目
        """
        return self.create_sequence_entry(
            seq_type="ion",
            ion=ion,
            count=count
        )
    
    def create_structure_json(self,
                             sequences: List[Dict[str, Any]],
                             name: str = "",
                             covalent_bonds: Optional[List[Dict[str, Any]]] = None,
                             assembly_id: Optional[str] = None) -> Dict[str, Any]:
        """
        创建完整的结构JSON数据
        
        Parameters
        ----------
        sequences : List[Dict[str, Any]]
            序列条目列表
        name : str
            结构名称
        covalent_bonds : Optional[List[Dict[str, Any]]]
            共价键信息列表
        assembly_id : Optional[str]
            Assembly ID
            
        Returns
        -------
        Dict[str, Any]
            完整的JSON数据字典
        """
        data = {
            "sequences": sequences,
            "name": name
        }
        
        if covalent_bonds:
            data["covalent_bonds"] = covalent_bonds
        
        if assembly_id:
            data["assembly_id"] = assembly_id
        
        return data
    
    def write_structure(self,
                       sequences: List[Dict[str, Any]],
                       path: Union[str, Path],
                       name: str = "",
                       covalent_bonds: Optional[List[Dict[str, Any]]] = None,
                       assembly_id: Optional[str] = None) -> Path:
        """
        写入结构JSON文件
        
        Parameters
        ----------
        sequences : List[Dict[str, Any]]
            序列条目列表
        path : Union[str, Path]
            输出文件路径
        name : str
            结构名称
        covalent_bonds : Optional[List[Dict[str, Any]]]
            共价键信息列表
        assembly_id : Optional[str]
            Assembly ID
            
        Returns
        -------
        Path
            写入的文件路径
        """
        data = self.create_structure_json(
            sequences=sequences,
            name=name,
            covalent_bonds=covalent_bonds,
            assembly_id=assembly_id
        )
        return self.write(data, path, name=name)
