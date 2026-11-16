"""
特征处理管道

统一的生物学特征处理管道，参考Protenix和OpenFold实现
"""

from typing import Dict, Any, Optional, List
import numpy as np

from onescience.datapipes.biology.common.features.feature_base import (
    FeatureDict,
    FeaturePipeline,
)
from onescience.datapipes.biology.common.features.sequence_features import (
    make_sequence_features,
    create_target_feat,
)
from onescience.datapipes.biology.common.features.msa_features import (
    make_msa_features,
    create_msa_feat,
    sample_msa,
)
from onescience.datapipes.biology.common.features.structure_features import (
    make_structure_features,
    make_pseudo_beta,
)
from onescience.datapipes.biology.common.features.feature_utils import (
    cast_to_64bit_ints,
    squeeze_features,
    pad_features,
    crop_features,
    merge_features,
)


class BiologyFeaturePipeline(FeaturePipeline):
    """
    生物学特征处理管道
    
    统一处理序列、MSA、结构等特征，支持多种模型输入格式
    """
    
    def __init__(
        self,
        config: Dict[str, Any] = None,
        use_msa: bool = True,
        use_structure: bool = False,
        max_msa_seqs: Optional[int] = None,
        crop_size: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        config : Dict[str, Any]
            配置字典
        use_msa : bool
            是否使用MSA特征
        use_structure : bool
            是否使用结构特征
        max_msa_seqs : Optional[int]
            最大MSA序列数
        crop_size : Optional[int]
            裁剪大小
        """
        super().__init__(config)
        self.use_msa = use_msa
        self.use_structure = use_structure
        self.max_msa_seqs = max_msa_seqs
        self.crop_size = crop_size
    
    def process(self, raw_features: FeatureDict) -> FeatureDict:
        """
        处理原始特征
        
        Parameters
        ----------
        raw_features : FeatureDict
            原始特征字典
            
        Returns
        -------
        FeatureDict
            处理后的特征字典
        """
        features = raw_features.copy()
        
        # 1. 数据类型转换
        features = cast_to_64bit_ints(features)
        
        # 2. 压缩维度
        features = squeeze_features(features)
        
        # 3. 创建组合特征
        features = self._create_composite_features(features)
        
        # 4. 处理MSA
        if self.use_msa and "msa" in features:
            features = self._process_msa(features)
        
        # 5. 处理结构
        if self.use_structure and "all_atom_positions" in features:
            features = self._process_structure(features)
        
        # 6. 裁剪
        if self.crop_size is not None:
            features = self._crop_features(features)
        
        return features
    
    def _create_composite_features(self, features: FeatureDict) -> FeatureDict:
        """
        创建组合特征
        
        Parameters
        ----------
        features : FeatureDict
            特征字典
            
        Returns
        -------
        FeatureDict
            更新后的特征字典
        """
        # 创建target特征
        if "aatype" in features:
            between_segment = features.get("between_segment_residues", None)
            features["target_feat"] = create_target_feat(
                aatype=features["aatype"],
                between_segment_residues=between_segment,
            )
        
        # 创建MSA特征
        if "msa" in features and "deletion_matrix" in features:
            features["msa_feat"] = create_msa_feat(
                msa=features["msa"],
                deletion_matrix=features["deletion_matrix"],
            )
        
        return features
    
    def _process_msa(self, features: FeatureDict) -> FeatureDict:
        """
        处理MSA特征
        
        Parameters
        ----------
        features : FeatureDict
            特征字典
            
        Returns
        -------
        FeatureDict
            更新后的特征字典
        """
        if self.max_msa_seqs and "msa" in features:
            # 采样MSA
            msa_features = {k: v for k, v in features.items() 
                          if k in ["msa", "deletion_matrix", "msa_mask", "msa_row_mask"]}
            sampled = sample_msa(msa_features, self.max_msa_seqs)
            features.update(sampled)
        
        return features
    
    def _process_structure(self, features: FeatureDict) -> FeatureDict:
        """
        处理结构特征
        
        Parameters
        ----------
        features : FeatureDict
            特征字典
            
        Returns
        -------
        FeatureDict
            更新后的特征字典
        """
        # 添加伪β碳特征
        features = make_pseudo_beta(features)
        
        return features
    
    def _crop_features(self, features: FeatureDict) -> FeatureDict:
        """
        裁剪特征
        
        Parameters
        ----------
        features : FeatureDict
            特征字典
            
        Returns
        -------
        FeatureDict
            裁剪后的特征字典
        """
        if self.crop_size is None:
            return features
        
        # 获取序列长度
        seq_length = features.get("seq_length", 0)
        if seq_length <= self.crop_size:
            return features
        
        # 随机裁剪起始位置
        crop_start = np.random.randint(0, seq_length - self.crop_size + 1)
        
        # 裁剪特征
        cropped = crop_features(features, crop_start, self.crop_size)
        
        # 更新序列长度
        cropped["seq_length"] = np.array(self.crop_size, dtype=np.int32)
        
        return cropped
    
    def process_sequence(
        self,
        sequence: str,
        sequence_type: str = "protein",
    ) -> FeatureDict:
        """
        处理单个序列
        
        Parameters
        ----------
        sequence : str
            序列字符串
        sequence_type : str
            序列类型
            
        Returns
        -------
        FeatureDict
            序列特征字典
        """
        return make_sequence_features(
            sequence=sequence,
            sequence_type=sequence_type,
        )
    
    def process_msa(
        self,
        sequences: List[str],
        deletion_matrix: Optional[List[List[int]]] = None,
    ) -> FeatureDict:
        """
        处理MSA
        
        Parameters
        ----------
        sequences : List[str]
            MSA序列列表
        deletion_matrix : Optional[List[List[int]]]
            删除矩阵
            
        Returns
        -------
        FeatureDict
            MSA特征字典
        """
        return make_msa_features(
            sequences=sequences,
            deletion_matrix=deletion_matrix,
            max_seqs=self.max_msa_seqs,
        )
    
    def process_structure(
        self,
        positions: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> FeatureDict:
        """
        处理结构
        
        Parameters
        ----------
        positions : np.ndarray
            原子坐标
        mask : Optional[np.ndarray]
            原子掩码
            
        Returns
        -------
        FeatureDict
            结构特征字典
        """
        return make_structure_features(
            positions=positions,
            mask=mask,
        )


def np_example_to_features(
    np_example: FeatureDict,
    config: Dict[str, Any],
    mode: str = "train",
) -> FeatureDict:
    """
    将numpy示例转换为特征
    
    参考OpenFold的np_example_to_features实现
    
    Parameters
    ----------
    np_example : FeatureDict
        numpy示例字典
    config : Dict[str, Any]
        配置字典
    mode : str
        模式: "train", "eval", "predict"
        
    Returns
    -------
    FeatureDict
        处理后的特征字典
    """
    features = np_example.copy()
    
    # 创建处理管道
    pipeline = BiologyFeaturePipeline(
        config=config,
        use_msa=config.get("use_msa", True),
        use_structure=config.get("use_structure", False),
        max_msa_seqs=config.get("max_msa_seqs", None),
        crop_size=config.get("crop_size", None),
    )
    
    # 处理特征
    features = pipeline.process(features)
    
    return features


def make_data_config(
    config: Dict[str, Any],
    mode: str,
    num_res: int,
) -> tuple:
    """
    创建数据配置
    
    参考OpenFold的make_data_config实现
    
    Parameters
    ----------
    config : Dict[str, Any]
        配置字典
    mode : str
        模式
    num_res : int
        残基数
        
    Returns
    -------
    tuple
        (配置, 特征名称列表)
    """
    cfg = config.copy()
    
    # 设置裁剪大小
    if cfg.get("crop_size") is None:
        cfg["crop_size"] = num_res
    
    # 确定特征名称
    feature_names = cfg.get("unsupervised_features", [])
    
    if cfg.get("use_templates", False):
        feature_names += cfg.get("template_features", [])
    
    if mode == "train" and cfg.get("supervised", False):
        feature_names += cfg.get("supervised_features", [])
    
    return cfg, feature_names


class UnifiedFeaturePipeline:
    """
    统一特征处理管道
    
    为不同的生物学模型（Protenix, OpenFold等）提供统一的特征处理接口
    """
    
    def __init__(
        self,
        model_type: str = "generic",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Parameters
        ----------
        model_type : str
            模型类型: "generic", "protenix", "openfold"
        config : Optional[Dict[str, Any]]
            配置字典
        """
        self.model_type = model_type.lower()
        self.config = config or {}
        
        # 创建基础管道
        self.base_pipeline = BiologyFeaturePipeline(
            config=self.config,
            use_msa=self.config.get("use_msa", True),
            use_structure=self.config.get("use_structure", False),
            max_msa_seqs=self.config.get("max_msa_seqs", None),
        )
    
    def process(self, raw_data: Dict[str, Any]) -> FeatureDict:
        """
        处理原始数据
        
        Parameters
        ----------
        raw_data : Dict[str, Any]
            原始数据字典
            
        Returns
        -------
        FeatureDict
            处理后的特征字典
        """
        features = {}
        
        # 处理序列
        if "sequence" in raw_data:
            seq_features = self.base_pipeline.process_sequence(
                sequence=raw_data["sequence"],
                sequence_type=raw_data.get("sequence_type", "protein"),
            )
            features.update(seq_features)
        
        # 处理MSA
        if "msa_sequences" in raw_data and self.config.get("use_msa", True):
            msa_features = self.base_pipeline.process_msa(
                sequences=raw_data["msa_sequences"],
                deletion_matrix=raw_data.get("deletion_matrix", None),
            )
            features.update(msa_features)
        
        # 处理结构
        if "positions" in raw_data and self.config.get("use_structure", False):
            struct_features = self.base_pipeline.process_structure(
                positions=raw_data["positions"],
                mask=raw_data.get("mask", None),
            )
            features.update(struct_features)
        
        # 使用基础管道进一步处理
        features = self.base_pipeline.process(features)
        
        # 模型特定的后处理
        features = self._model_specific_processing(features)
        
        return features
    
    def _model_specific_processing(self, features: FeatureDict) -> FeatureDict:
        """
        模型特定的后处理
        
        Parameters
        ----------
        features : FeatureDict
            特征字典
            
        Returns
        -------
        FeatureDict
            处理后的特征字典
        """
        if self.model_type == "protenix":
            # Protenix特定处理
            features = self._protenix_processing(features)
        elif self.model_type == "openfold":
            # OpenFold特定处理
            features = self._openfold_processing(features)
        
        return features
    
    def _protenix_processing(self, features: FeatureDict) -> FeatureDict:
        """
        Protenix特定处理
        
        Parameters
        ----------
        features : FeatureDict
            特征字典
            
        Returns
        -------
        FeatureDict
            处理后的特征字典
        """
        # Protenix特定的特征转换
        # 例如：aatype从整数转换为one-hot
        if "aatype" in features and features["aatype"].ndim == 1:
            from onescience.datapipes.biology.common.features.sequence_features import (
                restype_onehot_encode,
            )
            features["aatype"] = restype_onehot_encode(
                sequence="".join(["A"] * len(features["aatype"])),
                num_classes=32,  # Protenix使用32类
            )
        
        return features
    
    def _openfold_processing(self, features: FeatureDict) -> FeatureDict:
        """
        OpenFold特定处理
        
        Parameters
        ----------
        features : FeatureDict
            特征字典
            
        Returns
        -------
        FeatureDict
            处理后的特征字典
        """
        # OpenFold特定的特征转换
        # 例如：确保aatype是整数类型
        if "aatype" in features and features["aatype"].ndim > 1:
            features["aatype"] = np.argmax(features["aatype"], axis=-1)
        
        return features
