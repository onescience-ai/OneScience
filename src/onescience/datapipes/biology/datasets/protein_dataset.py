"""
统一的蛋白质数据集基类

提供基础的数据处理能力，用户可以继承并实现自己的适配逻辑。
不强制依赖 adapter，用户可以选择使用 adapter 或自己实现。
"""

import re
import logging
import time
import traceback
import warnings

from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from onescience.datapipes.biology import BioDataset
from onescience.datapipes.core.config import DatasetConfig
from onescience.datapipes.biology.common.sequence.fasta_parser import FASTAParser
from onescience.datapipes.biology.common.json.json_parser import JSONParser
from onescience.datapipes.biology.common.sequence.sequence_encoder import (
    AminoAcidEncoder,
)
from onescience.datapipes.biology.common.msa.msa_parser import MSAParser
from onescience.datapipes.biology.common.msa.msa_featurizer import MSAFeaturizer
from onescience.datapipes.biology.common.utils.file_utils import (
    detect_file_format,
    get_all_fasta_files,
    get_all_msa_files,
    get_all_structure_files,
)
from onescience.datapipes.biology.datasets.unified_dataset import UnifiedDataPipeline

import json


logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", module="biotite")


class ProteinDataset(BioDataset):
    """
    统一的蛋白质数据集基类
    
    提供基础的数据处理能力，用户可以：
    1. 直接继承并实现自己的 `__getitem__` 方法
    2. 使用 `UnifiedDataPipeline` 进行数据处理
    3. 可选使用 adapter（通过 `use_adapter=True`）
    
    设计理念：
    - 提供基础组件，不强制依赖 adapter
    - 用户完全控制数据处理流程
    - 通过 pipeline 或自定义方法实现适配
    
    Examples
    --------
    # 方式1: 直接继承，自己实现适配逻辑
    >>> class MyProteinDataset(ProteinDataset):
    ...     def __getitem__(self, idx):
    ...         sample = self.data_list[idx]
    ...         # 使用 pipeline 处理
    ...         features = self.pipeline.process_sample(
    ...             sequence=sample.get("sequence"),
    ...             msa_path=sample.get("msa_path")
    ...         )
    ...         # 自己实现适配逻辑
    ...         return self._adapt_to_my_model(features)
    
    # 方式2: 使用可选的 adapter
    >>> config = {
    ...     "source": {"path": "/data/proteins"},
    ...     "data": {
    ...         "extra": {
    ...             "model_name": "protenix",
    ...             "use_adapter": True  # 启用 adapter
    ...         }
    ...     }
    ... }
    >>> dataset = ProteinDataset(config)
    """
    
    def __init__(self, config: Union[DatasetConfig, Dict[str, Any]]):
        super().__init__(config)

        # 初始化通用处理模块（基础组件）
        self.json_parser = JSONParser()
        self.fasta_parser = FASTAParser()
        self.aa_encoder = AminoAcidEncoder()
        self.msa_parser = MSAParser()
        self.msa_featurizer = MSAFeaturizer(
            max_seqs=self.config.data.extra.get('max_msa_seqs')
        )

        # 初始化统一数据管道（可选，用户可以选择使用）
        use_pipeline = self.config.data.extra.get('use_pipeline', True)
        if use_pipeline:
            self.pipeline = UnifiedDataPipeline(
                use_msa=self.config.data.extra.get('use_msa', True),
                use_structure=self.config.data.extra.get('use_structure', False),
                max_msa_seqs=self.config.data.extra.get('max_msa_seqs'),
            )
        else:
            self.pipeline = None

        # 可选的 adapter 支持（不强制）
        self.adapter = None
        use_adapter = self.config.data.extra.get('use_adapter', False)
        if use_adapter:
            try:
                from onescience.datapipes.biology.adapters import get_adapter
                model_name = self.config.data.extra.get("model_name", "protenix")
                self.adapter = get_adapter(model_name, self.config)
                self.logger.info(f"Using adapter: {model_name}")
            except ImportError:
                self.logger.warning("Adapter module not available, continuing without adapter")

        # # 初始化数据列表
        # self._init_data()
    
    def _init_data(self):
        """初始化数据"""
        # 加载数据索引
        self.data_list = self._load_data_list()
        self.logger.info(f"Loaded {len(self.data_list)} samples")
    
    @staticmethod
    def _strip_all_suffixes(path: Path) -> str:
        """
        移除文件名中的所有扩展名（包括压缩后缀），返回纯粹的基名。
        """
        name = path.name
        for suffix in path.suffixes:
            name = name[: -len(suffix)]
        return name or path.stem
    
    @staticmethod
    def _normalize_identifier(candidate: Optional[str], fallback: str) -> str:
        """
        归一化序列描述或文件名以生成稳定的样本标识符。
        """
        if candidate:
            token = candidate.strip()
            if token:
                token = token.split()[0]
                token = token.split("|")[0]
                token = re.sub(r"[^0-9A-Za-z]+", "_", token).strip("_")
                if token:
                    return token.lower()
        clean_fallback = re.sub(r"[^0-9A-Za-z]+", "_", fallback).strip("_")
        return (clean_fallback or fallback).lower()
    
    def _load_data_list(self) -> List[Dict[str, Any]]:
        """
        加载数据索引列表
        
        子类可以重写此方法以实现自定义的数据加载逻辑
        
        Returns
        -------
        List[Dict[str, Any]]
            数据索引列表，每个元素包含：
            - sequence: 序列字符串（可选）
            - structure_path: 结构文件路径（可选）
            - msa_path: MSA文件路径（可选）
        """
        data_list: List[Dict[str, Any]] = []
        data_path = Path(self.data_path)
        extra_cfg: Dict[str, Any] = getattr(self.config.data, "extra", {}) or {}
        recursive = bool(extra_cfg.get("recursive", False))
        allow_dummy = extra_cfg.get("allow_dummy_sample", True)
        default_chain_id: Optional[str] = extra_cfg.get("default_chain_id")
        structure_dirs_config = extra_cfg.get("structure_dirs")
        msa_dirs_config = extra_cfg.get("msa_dirs")
        
        entries: Dict[str, Dict[str, Any]] = {}
        
        def ensure_entry(key: str) -> Dict[str, Any]:
            if key not in entries:
                entries[key] = {"id": key}
            return entries[key]
        
        def append_sequence_entry(
            fasta_file: Path,
            sequence: str,
            description: Optional[str],
            seq_index: int,
        ) -> None:
            stem = self._strip_all_suffixes(fasta_file)
            key = self._normalize_identifier(description, f"{stem}_{seq_index}")
            entry = ensure_entry(key)
            if sequence and not entry.get("sequence"):
                entry["sequence"] = sequence
            if description and not entry.get("description"):
                entry["description"] = description
            entry.setdefault("sequence_index", seq_index)
            entry.setdefault("fasta_path", str(fasta_file))
        
        def attach_msa(msa_file: Path) -> None:
            format_hint = detect_file_format(msa_file)
            stem = self._strip_all_suffixes(msa_file)
            key = self._normalize_identifier(stem, stem)
            entry = entries.get(key)
            if entry is None:
                # 尝试基于描述或文件名前缀的松散匹配
                normalize_candidates = [
                    key,
                    self._normalize_identifier(stem.split("_")[0], stem),
                ]
                for candidate in normalize_candidates:
                    if candidate in entries:
                        entry = entries[candidate]
                        key = candidate
                        break
            if entry is None:
                try:
                    msa = self.msa_parser.parse_file(msa_file, format=format_hint)
                    if len(msa.sequences) == 0:
                        self.logger.warning(f"MSA文件 {msa_file} 未包含任何序列，跳过关联")
                        return
                    description = msa.descriptions[0] if msa.descriptions else stem
                    key = self._normalize_identifier(description, stem)
                    entry = ensure_entry(key)
                    if not entry.get("sequence"):
                        entry["sequence"] = msa.sequences[0]
                        entry["description"] = description
                except Exception as exc:
                    self.logger.warning(f"解析 MSA 文件失败: {msa_file} ({exc})")
                    return
            entry["msa_path"] = str(msa_file)
            if format_hint:
                entry.setdefault("msa_format", format_hint)
        
        def attach_structure(structure_file: Path) -> None:
            format_hint = detect_file_format(structure_file)
            stem = self._strip_all_suffixes(structure_file)
            key = self._normalize_identifier(stem, stem)
            entry = entries.get(key)
            if entry is None:
                # 结构文件往往包含链信息，例如 1abc_A
                normalized_stem = self._normalize_identifier(stem.split("_")[0], stem)
                entry = entries.get(normalized_stem)
                if entry:
                    key = normalized_stem
            if entry is None:
                entry = ensure_entry(key)
            entry["structure_path"] = str(structure_file)
            if default_chain_id and "chain_id" not in entry:
                entry["chain_id"] = default_chain_id
            if format_hint:
                entry.setdefault("structure_format", format_hint)
        
        # 根据配置加载数据
        if data_path.is_file():
            # 如果是文件，检测格式并解析
            file_format = detect_file_format(data_path)
            if file_format == 'fasta':
                sequences, descriptions = self.fasta_parser.parse_file(data_path)
                for idx, (seq, desc) in enumerate(zip(sequences, descriptions)):
                    append_sequence_entry(data_path, seq, desc, idx)
            elif file_format in {'a3m', 'stockholm', 'clustal', 'phylip'}:
                attach_msa(data_path)
            elif file_format in {'pdb', 'mmcif', 'mmtf'}:
                attach_structure(data_path)
            else:
                self.logger.warning(f"Unsupported file format: {file_format} for {data_path}")
        elif data_path.is_dir():
            # 如果是目录，查找所有相关文件（包括压缩文件）
            # 查找FASTA文件（支持多种扩展名和压缩格式）
            fasta_files = get_all_fasta_files(data_path, recursive=recursive)
            for fasta_file in fasta_files:
                try:
                    sequences, descriptions = self.fasta_parser.parse_file(fasta_file)
                except Exception as exc:
                    self.logger.warning(f"解析 FASTA 文件失败: {fasta_file} ({exc})")
                    continue
                for idx, (seq, desc) in enumerate(zip(sequences, descriptions)):
                    append_sequence_entry(fasta_file, seq, desc, idx)
            
            # 查找MSA文件（支持多种格式和压缩）
            msa_files = get_all_msa_files(data_path, recursive=recursive)
            for msa_file in msa_files:
                attach_msa(msa_file)
            
            # 查找结构文件
            structure_files = get_all_structure_files(data_path, recursive=recursive)
            for structure_file in structure_files:
                attach_structure(structure_file)
        else:
            self.logger.warning(f"数据路径既不是文件也不是目录: {data_path}")
        
        # 额外的MSA目录（如果配置给出）
        msa_dirs: List[Path] = []
        if msa_dirs_config:
            if isinstance(msa_dirs_config, (str, Path)):
                msa_dirs.append(Path(msa_dirs_config))
            else:
                msa_dirs.extend(Path(p) for p in msa_dirs_config)
        if getattr(self, "msa_path", None):
            msa_dirs.append(Path(self.msa_path))
        
        for msa_dir in msa_dirs:
            if not msa_dir.exists():
                self.logger.warning(f"配置中的 MSA 目录不存在: {msa_dir}")
                continue
            if msa_dir.is_file():
                attach_msa(msa_dir)
            else:
                for msa_file in get_all_msa_files(msa_dir, recursive=recursive):
                    attach_msa(msa_file)
        
        # 额外的结构目录（如果配置给出）
        structure_dirs: List[Path] = []
        if structure_dirs_config:
            if isinstance(structure_dirs_config, (str, Path)):
                structure_dirs.append(Path(structure_dirs_config))
            else:
                structure_dirs.extend(Path(p) for p in structure_dirs_config)
        
        for struct_dir in structure_dirs:
            if not struct_dir.exists():
                self.logger.warning(f"配置中的结构目录不存在: {struct_dir}")
                continue
            if struct_dir.is_file():
                attach_structure(struct_dir)
            else:
                for struct_file in get_all_structure_files(struct_dir, recursive=recursive):
                    attach_structure(struct_file)
        
        # 收集最终数据列表
        if entries:
            data_list = list(entries.values())
        
        # 如果数据列表为空，创建一个虚拟样本用于测试
        if not data_list and allow_dummy:
            self.logger.warning("No data found, creating dummy sample")
            data_list.append({
                "id": "dummy_sample",
                "sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL",
            })
        
        return data_list
    
    def __getitem__(self, index: int) -> tuple[Dict[str, Any], ...]:
        """
        获取样本
        
        默认实现：如果启用了 adapter, 使用 adapter 处理；否则使用 pipeline。
        子类应该重写此方法以实现自己的适配逻辑。
        
        Parameters
        ----------
        idx : int
            样本索引
            
        Returns
        -------
        tuple[Dict[str, Any], ...]
            特征字典
        """
        # if not (-len(self) <= index < len(self)):
        #     raise IndexError(f"Index {index} out of range for dataset of size {len(self)}")
        # sample = self.data_list[index % len(self)]

        if self.config.input_json_path is not None:
            with open(self.config.input_json_path, "r") as f:
                self.inputs = json.load(f)
            single_sample_dict = self.inputs[index]
            sample_name = single_sample_dict["name"]
            logger.info(f"Featurizing {sample_name}...")
            # 如果使用 adapter，通过 adapter 处理
            if self.adapter is not None:
                adapter_name = self.config.data.extra.get("model_name", "protenix")
                if adapter_name == "protenix_infer_adapter":
                    try:
                        features_dict, atom_array, _ = self.adapter.process_json_sample(single_sample_dict)
                        error_message = ""
                    except Exception as e:
                        features_dict, atom_array = {}, None
                        error_message = f"{e}:\n{traceback.format_exc()}"
                    # 构建与 biology_inference.py 兼容的返回格式
                    result = {
                        "features": features_dict,
                        "sample_name": single_sample_dict["name"],
                        "sample_index": index,
                    }
                    return result, atom_array, error_message
                elif adapter_name == "openfold":
                    pass
                else:
                    pass
            if self.pipeline is not None:
                features = self.pipeline.process_sample(
                    sequence=single_sample_dict.get("sequence"),
                    fasta_path=Path(single_sample_dict["fasta_path"]) if "fasta_path" in single_sample_dict else None,
                    msa_path=Path(single_sample_dict["msa_path"]) if "msa_path" in single_sample_dict else None,
                    structure_path=Path(single_sample_dict["structure_path"]) if     "structure_path" in single_sample_dict else None,)
                return features
        elif self.config.fase_path is not None:
            pass
        else:
            pass
        
        # 如果都没有，返回原始样本（用户需要自己处理）
        return single_sample_dict
    
    def __len__(self) -> int:
        """返回数据集大小"""
        if self.config.input_json_path is not None:
            with open(self.config.input_json_path, "r") as f:
                self.inputs = json.load(f)
            return len(self.inputs)
        else: # TODO
            return None

