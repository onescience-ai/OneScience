"""
统一的基因组数据集基类

提供基础的数据处理能力，用户可以继承并实现自己的适配逻辑。
不强制依赖 adapter，用户可以选择使用 adapter 或自己实现。
"""

from typing import Any, Dict, List, Union
from pathlib import Path

from onescience.datapipes.biology import BioDataset
from onescience.datapipes.core.config import DatasetConfig
from onescience.datapipes.biology.common.sequence.fasta_parser import FASTAParser
from onescience.datapipes.biology.common.sequence.sequence_encoder import (
    NucleotideEncoder,
)
from onescience.datapipes.biology.common.utils.file_utils import (
    detect_file_format,
    get_all_fasta_files,
)
from onescience.datapipes.biology.datasets.unified_dataset import UnifiedDataPipeline


class GenomeDataset(BioDataset):
    """
    统一的基因组数据集基类
    
    提供基础的数据处理能力，用户可以：
    1. 直接继承并实现自己的 `__getitem__` 方法
    2. 使用 `UnifiedDataPipeline` 进行数据处理
    3. 可选使用 adapter（通过 `use_adapter=True`）

    Examples
    --------
    # 方式1: 直接继承，自己实现适配逻辑
    >>> class MyGenomeDataset(GenomeDataset):
    ...     def __getitem__(self, idx):
    ...         sample = self.data_list[idx]
    ...         # 使用 pipeline 处理
    ...         features = self.pipeline.process_sample(
    ...             sequence=sample.get("sequence")
    ...         )
    ...         # 自己实现适配逻辑
    ...         return self._adapt_to_my_model(features)
    
    # 方式2: 使用可选的 adapter
    >>> config = {
    ...     "source": {"path": "/data/genomes"},
    ...     "data": {
    ...         "extra": {
    ...             "model_name": "evo2",
    ...             "sequence_type": "DNA",
    ...             "use_adapter": True  # 启用 adapter
    ...         }
    ...     }
    ... }
    >>> dataset = GenomeDataset(config)
    """
    
    def __init__(self, config: Union[DatasetConfig, Dict[str, Any]]):
        super().__init__(config)
        
        # 初始化通用处理模块（基础组件）
        self.fasta_parser = FASTAParser()
        sequence_type = self.config.data.extra.get("sequence_type", "DNA")
        self.nt_encoder = NucleotideEncoder(sequence_type=sequence_type)
        
        # 初始化统一数据管道（可选，用户可以选择使用）
        use_pipeline = self.config.data.extra.get('use_pipeline', True)
        if use_pipeline:
            self.pipeline = UnifiedDataPipeline(
                use_msa=self.config.data.extra.get('use_msa', False),
                use_structure=False,  # 基因组数据通常不需要结构
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
                model_name = self.config.data.extra.get("model_name", "evo2")
                self.adapter = get_adapter(model_name, self.config)
                self.logger.info(f"Using adapter: {model_name}")
            except ImportError:
                self.logger.warning("Adapter module not available, continuing without adapter")
    
    def _init_data(self):
        """初始化数据"""
        self.data_list = self._load_data_list()
        self.logger.info(f"Loaded {len(self.data_list)} genome sequences")
    
    def _load_data_list(self) -> List[Dict[str, Any]]:
        """
        加载基因组数据索引
        
        子类可以重写此方法以实现自定义的数据加载逻辑
        
        Returns
        -------
        List[Dict[str, Any]]
            数据索引列表
        """
        data_list = []
        data_path = Path(self.data_path)
        
        if data_path.is_file():
            file_format = detect_file_format(data_path)
            if file_format == 'fasta':
                sequences, descriptions = self.fasta_parser.parse_file(data_path)
                for seq, desc in zip(sequences, descriptions):
                    data_list.append({
                        "sequence": seq,
                        "description": desc,
                    })
        elif data_path.is_dir():
            # 使用工具函数获取所有FASTA文件（包括压缩文件）
            fasta_files = get_all_fasta_files(data_path, recursive=False)
            for fasta_file in fasta_files:
                sequences, descriptions = self.fasta_parser.parse_file(fasta_file)
                for seq, desc in zip(sequences, descriptions):
                    data_list.append({
                        "sequence": seq,
                        "description": desc,
                        "file_path": str(fasta_file),
                    })
        
        return data_list
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取样本
        
        默认实现：如果启用了 adapter，使用 adapter 处理；否则使用 pipeline。
        子类应该重写此方法以实现自己的适配逻辑。
        
        Parameters
        ----------
        idx : int
            样本索引
            
        Returns
        -------
        Dict[str, Any]
            特征字典
        """
        sample = self.data_list[idx]
        
        # 如果使用 adapter，通过 adapter 处理
        if self.adapter is not None:
            features = self.adapter.process_sample(sample)
            return features
        
        # 否则使用 pipeline 处理
        if self.pipeline is not None:
            features = self.pipeline.process_sample(
                sequence=sample.get("sequence"),
                fasta_path=Path(sample["file_path"]) if "file_path" in sample else None,
            )
            return features
        
        return sample
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data_list)

