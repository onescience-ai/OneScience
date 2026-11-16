"""
生物学特征处理模块使用示例

展示如何使用新实现的特征处理模块
"""

import numpy as np

# ==============================================================================
# 示例1: 基本序列特征提取
# ==============================================================================

def example_sequence_features():
    """序列特征提取示例"""
    from onescience.datapipes.biology.common.features import (
        SequenceFeatureExtractor,
        make_sequence_features,
    )
    
    # 方法1: 使用特征提取器类
    extractor = SequenceFeatureExtractor(sequence_type="protein")
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    features = extractor.extract(sequence)
    
    print("序列特征:")
    for key, value in features.items():
        print(f"  {key}: {value.shape if hasattr(value, 'shape') else value}")
    
    # 方法2: 使用函数直接提取
    features = make_sequence_features(sequence, sequence_type="protein")
    
    return features


# ==============================================================================
# 示例2: MSA特征提取
# ==============================================================================

def example_msa_features():
    """MSA特征提取示例"""
    from onescience.datapipes.biology.common.features import (
        MSAFeatureExtractor,
        make_msa_features,
    )
    
    # 示例MSA数据
    msa_sequences = [
        "ACDEFGHIKLMNPQRSTVWY",  # 主序列
        "ACD-FGHIKLMNPQRSTVWY",  # 有gap的序列
        "ACDEFGHIKLMN-QRSTVWY",
        "ACDEFGH-KLMNPQRSTVWY",
    ]
    
    deletion_matrix = [
        [0] * 20,
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    
    # 方法1: 使用特征提取器类
    extractor = MSAFeatureExtractor(max_seqs=512)
    msa_data = {
        "sequences": msa_sequences,
        "deletion_matrix": deletion_matrix,
    }
    features = extractor.extract(msa_data)
    
    print("\nMSA特征:")
    for key, value in features.items():
        print(f"  {key}: {value.shape if hasattr(value, 'shape') else value}")
    
    # 方法2: 使用函数直接提取
    features = make_msa_features(
        sequences=msa_sequences,
        deletion_matrix=deletion_matrix,
        max_seqs=512,
    )
    
    return features


# ==============================================================================
# 示例3: 结构特征提取
# ==============================================================================

def example_structure_features():
    """结构特征提取示例"""
    from onescience.datapipes.biology.common.features import (
        StructureFeatureExtractor,
        make_structure_features,
    )
    
    # 示例结构数据 (N_res, N_atoms, 3)
    num_res = 100
    num_atoms = 37  # atom37表示
    positions = np.random.randn(num_res, num_atoms, 3).astype(np.float32)
    mask = np.random.randint(0, 2, size=(num_res, num_atoms)).astype(np.float32)
    
    # 方法1: 使用特征提取器类
    extractor = StructureFeatureExtractor(
        atom_types=['CA', 'C', 'N', 'O', 'CB'],
        compute_frames=True,
    )
    struct_data = {
        "positions": positions,
        "mask": mask,
    }
    features = extractor.extract(struct_data)
    
    print("\n结构特征:")
    for key, value in features.items():
        print(f"  {key}: {value.shape if hasattr(value, 'shape') else value}")
    
    # 方法2: 使用函数直接提取
    features = make_structure_features(
        positions=positions,
        mask=mask,
        compute_frames=True,
    )
    
    return features


# ==============================================================================
# 示例4: 使用特征处理管道
# ==============================================================================

def example_feature_pipeline():
    """特征处理管道示例"""
    from onescience.datapipes.biology.common.features import (
        BiologyFeaturePipeline,
        UnifiedFeaturePipeline,
    )
    
    # 方法1: 使用基础管道
    pipeline = BiologyFeaturePipeline(
        use_msa=True,
        use_structure=True,
        max_msa_seqs=512,
        crop_size=None,
    )
    
    # 原始特征数据
    raw_features = {
        "sequence": "ACDEFGHIKLMNPQRSTVWY",
        "msa_sequences": [
            "ACDEFGHIKLMNPQRSTVWY",
            "ACD-FGHIKLMNPQRSTVWY",
        ],
        "positions": np.random.randn(20, 37, 3).astype(np.float32),
    }
    
    # 分别处理各个部分
    seq_features = pipeline.process_sequence(raw_features["sequence"])
    msa_features = pipeline.process_msa(raw_features["msa_sequences"])
    struct_features = pipeline.process_structure(raw_features["positions"])
    
    # 合并特征
    all_features = {**seq_features, **msa_features, **struct_features}
    processed_features = pipeline.process(all_features)
    
    print("\n管道处理后的特征:")
    for key, value in processed_features.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: {value.shape}")
    
    # 方法2: 使用统一管道（支持模型特定处理）
    unified_pipeline = UnifiedFeaturePipeline(
        model_type="generic",  # 或 "protenix", "openfold"
        config={
            "use_msa": True,
            "use_structure": True,
            "max_msa_seqs": 512,
        }
    )
    
    processed = unified_pipeline.process(raw_features)
    
    return processed_features


# ==============================================================================
# 示例5: 使用工具函数
# ==============================================================================

def example_utility_functions():
    """特征工具函数示例"""
    from onescience.datapipes.biology.common.features import (
        encode_to_onehot,
        pad_features,
        crop_features,
        merge_features,
        cast_to_64bit_ints,
        make_one_hot,
    )
    
    # 示例特征
    features = {
        "aatype": np.array([0, 1, 2, 3, 4], dtype=np.int32),
        "msa": np.random.randint(0, 22, size=(10, 20)).astype(np.int32),
        "positions": np.random.randn(20, 37, 3).astype(np.float32),
    }
    
    # 1. One-hot编码
    aatype_onehot = make_onehot(features["aatype"], num_classes=22)
    print(f"\nOne-hot编码: {aatype_onehot.shape}")
    
    # 2. 数据类型转换
    features_64 = cast_to_64bit_ints(features)
    print(f"int32转换为int64: {features_64['aatype'].dtype}")
    
    # 3. 特征填充
    target_shapes = {
        "aatype": (100,),
        "msa": (10, 100),
    }
    padded = pad_features(features, target_shapes)
    print(f"填充后的形状: {padded['aatype'].shape}")
    
    # 4. 特征裁剪
    cropped = crop_features(features, crop_start=5, crop_size=10, spatial_dim=0)
    print(f"裁剪后的形状: {cropped['aatype'].shape}")
    
    # 5. 合并特征
    features2 = {
        "extra_feature": np.random.randn(20).astype(np.float32),
    }
    merged = merge_features([features, features2])
    print(f"合并后的键: {list(merged.keys())}")
    
    return merged


# ==============================================================================
# 示例6: 完整工作流程
# ==============================================================================

def example_complete_workflow():
    """完整工作流程示例"""
    from onescience.datapipes.biology.common.features import (
        UnifiedFeaturePipeline,
        compute_feature_statistics,
        validate_features,
    )
    
    # 创建处理管道
    pipeline = UnifiedFeaturePipeline(
        model_type="generic",
        config={
            "use_msa": True,
            "use_structure": True,
            "max_msa_seqs": 512,
            "crop_size": None,
        }
    )
    
    # 原始数据（模拟从文件读取）
    raw_data = {
        "sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL",
        "sequence_type": "protein",
        "msa_sequences": [
            # 这里应该是从MSA文件读取的序列
            "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL",
            "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL",
        ],
        "positions": np.random.randn(200, 37, 3).astype(np.float32),  # 模拟结构坐标
    }
    
    # 处理特征
    features = pipeline.process(raw_data)
    
    # 验证特征
    required_keys = ["aatype", "seq_length"]
    is_valid, missing = validate_features(features, required_keys)
    print(f"\n特征验证: {'通过' if is_valid else '失败'}")
    if not is_valid:
        print(f"缺失的键: {missing}")
    
    # 计算统计信息
    stats = compute_feature_statistics(features)
    print("\n特征统计:")
    for key, stat in list(stats.items())[:5]:  # 只显示前5个
        print(f"  {key}: {stat}")
    
    return features


# ==============================================================================
# 运行所有示例
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("生物学特征处理模块使用示例")
    print("=" * 70)
    
    # 运行示例
    example_sequence_features()
    example_msa_features()
    example_structure_features()
    example_feature_pipeline()
    example_utility_functions()
    example_complete_workflow()
    
    print("\n" + "=" * 70)
    print("所有示例运行完成！")
    print("=" * 70)
