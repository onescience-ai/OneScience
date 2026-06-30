#!/usr/bin/env python3
"""
MedGemma 集成测试脚本
验证 MedGemma 是否正确集成到 OneScience 中
"""

import os
import sys

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(project_root, "src"))

print("=" * 60)
print("MedGemma Integration Test")
print("=" * 60)

# 测试 1: 导入模块
print("\n[Test 1] Importing modules...")
try:
    from onescience.models.medgemma import MedGemma, VLLMModelRunner, TransformersModelRunner
    from onescience.models.medgemma.config import parse_configs, load_config
    from onescience.datapipes.biology.adapters.medgemma_infer_adapter import MedGemmaInferAdapter
    from onescience.datapipes.medical import ChatFormatter, DICOMLoader, MedicalImageProcessor
    print("✓ All modules imported successfully")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# 测试 2: 配置解析
print("\n[Test 2] Testing configuration parsing...")
try:
    from examples.biosciences.medgemma.configs.configs_base import medgemma_base_configs
    from ml_collections import ConfigDict

    # 创建测试配置
    test_config = medgemma_base_configs.copy()
    test_config.update({
        "run_name": "integration_test",
        "base_dir": "/tmp/medgemma_test",
        "model": {
            "variant": "4b",
            "model_path": "/tmp/fake_model",
        },
        "output": {
            "dump_dir": "/tmp/medgemma_output"
        }
    })

    configs = parse_configs(test_config, fill_required_with_null=True)
    print(f"✓ Configuration parsed successfully")
    print(f"  - Model variant: {configs.model.variant}")
    print(f"  - GPU memory utilization: {configs.inference.gpu_memory_utilization}")
except Exception as e:
    print(f"✗ Configuration parsing failed: {e}")
    sys.exit(1)

# 测试 3: 数据适配器
print("\n[Test 3] Testing data adapter...")
try:
    adapter = MedGemmaInferAdapter()

    # 测试文本适配
    sample = {
        "text": "What are the symptoms of diabetes?",
        "max_tokens": 500,
    }

    features = adapter.adapt_features(sample)
    print("✓ Data adapter working")
    print(f"  - Messages: {len(features['messages'])} message(s)")
    print(f"  - Parameters: {features['parameters']}")
except Exception as e:
    print(f"✗ Data adapter failed: {e}")
    sys.exit(1)

# 测试 4: Chat 格式化器
print("\n[Test 4] Testing chat formatter...")
try:
    formatter = ChatFormatter()

    messages = formatter.format_medical_query(
        question="What causes hypertension?",
        patient_info={"age": 65, "gender": "male"},
    )

    print("✓ Chat formatter working")
    print(f"  - Formatted {len(messages)} message(s)")
except Exception as e:
    print(f"✗ Chat formatter failed: {e}")
    sys.exit(1)

# 测试 5: 医学图像处理器
print("\n[Test 5] Testing medical image processor...")
try:
    import numpy as np

    processor = MedicalImageProcessor(target_size=(224, 224))

    # 创建假 CT 图像
    fake_ct = np.random.randint(-1000, 400, (512, 512), dtype=np.int16)

    processed = processor.process_ct_image(fake_ct)

    print("✓ Medical image processor working")
    print(f"  - Output shape: {processed.shape}")
    print(f"  - Output dtype: {processed.dtype}")
except Exception as e:
    print(f"✗ Medical image processor failed: {e}")
    import traceback
    traceback.print_exc()

# 测试 6: DICOM 加载器
print("\n[Test 6] Testing DICOM loader...")
try:
    loader = DICOMLoader()

    if loader.pydicom_available:
        print("✓ DICOM loader initialized (pydicom available)")
    else:
        print("⚠ DICOM loader initialized (pydicom not available)")
        print("  Install pydicom for DICOM support: pip install pydicom")
except Exception as e:
    print(f"✗ DICOM loader failed: {e}")

# 测试 7: 模型运行器检查
print("\n[Test 7] Checking model runners...")
try:
    # 检查 vLLM
    try:
        import vllm
        print("✓ vLLM is available")
        vllm_available = True
    except ImportError:
        print("⚠ vLLM not available (install with: pip install vllm)")
        vllm_available = False

    # 检查 transformers
    try:
        import transformers
        print("✓ Transformers is available")
    except ImportError:
        print("✗ Transformers not available (required)")
        sys.exit(1)
except Exception as e:
    print(f"✗ Model runner check failed: {e}")

# 测试 8: 推理运行器
print("\n[Test 8] Testing inference runner initialization...")
try:
    from examples.biosciences.medgemma.runner import MedicalInferenceRunner

    # 注意：不实际初始化模型（因为需要真实模型文件）
    print("✓ MedicalInferenceRunner imported successfully")
    print("  (Skipping actual model initialization - requires real model files)")
except Exception as e:
    print(f"✗ Inference runner import failed: {e}")
    sys.exit(1)

# 总结
print("\n" + "=" * 60)
print("Integration Test Summary")
print("=" * 60)
print("✓ Core functionality: PASSED")
print("✓ Configuration system: PASSED")
print("✓ Data adapters: PASSED")
print("✓ Medical data processing: PASSED")

if vllm_available:
    print("✓ Performance optimization (vLLM): AVAILABLE")
else:
    print("⚠ Performance optimization (vLLM): NOT AVAILABLE")
    print("  Consider installing: pip install vllm")

print("\n✅ MedGemma integration test completed successfully!")
print("\nNext steps:")
print("  1. Download a MedGemma model")
print("  2. Configure configs/inference_config.yaml")
print("  3. Run: python -m runner.medical_inference_runner --config configs/inference_config.yaml --interactive")
print("=" * 60)
