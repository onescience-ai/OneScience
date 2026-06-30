# MedGemma 基础配置
# 遵循 Protenix 的配置模式

from onescience.models.protenix.config.extend_types import (
    GlobalConfigValue,
    ListValue,
    RequiredValue,
    ValueMaybeNone,
)

basic_configs = {
    "project": "MedGemma",
    "run_name": RequiredValue(str),
    "base_dir": RequiredValue(str),
    "seed": 42,
    "deterministic": False,
    "use_wandb": False,
    "load_checkpoint_path": "",
    "eval_only": True,  # MedGemma 主要用于推理
}

model_configs = {
    # Model settings
    "variant": RequiredValue(str),  # "4b" or "27b"
    "model_path": RequiredValue(str),  # 模型权重路径
    "tokenizer_path": ValueMaybeNone(None),  # Tokenizer 路径，默认与 model_path 相同
    "prompt_format": "chat",  # "chat" or "instruct"
    "is_multimodal": True,  # 4B 支持多模态，27B 仅文本
}

inference_configs = {
    # Inference settings
    "gpu_memory_utilization": 0.9,
    "max_model_len": ValueMaybeNone(None),  # 最大序列长度，None 表示使用模型默认值
    "tensor_parallel_size": 1,  # Tensor 并行大小
    "default_max_tokens": 500,  # 默认生成最大 token 数
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": ValueMaybeNone(None),
    "min_p": ValueMaybeNone(None),
    "batch_size": 1,
    "num_workers": 0,
    "use_vllm": True,  # 是否使用 vLLM（如果为 False，则使用 transformers）
}

data_configs = {
    # Data settings
    "input_json_path": ValueMaybeNone(None),  # 输入 JSON 文件路径
    "input_dir": ValueMaybeNone(None),  # 输入目录
    "image_input_width": 224,
    "image_input_height": 224,
    "max_parallel_download_workers": 4,
    "worker_download_parallelism": "THREAD",  # "THREAD" or "PROCESS"
    "use_msa": False,  # MedGemma 不使用 MSA（这是 Protenix 特有的）
}

output_configs = {
    # Output settings
    "dump_dir": RequiredValue(str),
    "save_predictions": True,
    "output_format": "json",  # "json" or "jsonl"
    "save_intermediate": False,
}

# 合并所有配置
medgemma_base_configs = {
    **basic_configs,
    "model": model_configs,
    "inference": inference_configs,
    "data": data_configs,
    "output": output_configs,
}
