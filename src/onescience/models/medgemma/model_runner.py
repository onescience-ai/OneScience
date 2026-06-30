# MedGemma vLLM 模型运行器
# 实现基于 vLLM 的推理引擎

import logging
from typing import Any, Dict, List, Optional, Set, Mapping
import numpy as np

logger = logging.getLogger(__name__)


class VLLMModelRunner:
    """
    基于 vLLM 的 MedGemma 模型运行器
    兼容 MedGemma serving_framework 的 ModelRunner 接口
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        tensor_parallel_size: int = 1,
        trust_remote_code: bool = True,
    ):
        """
        初始化 vLLM 模型运行器

        Args:
            model_path: 模型权重路径
            tokenizer_path: Tokenizer 路径（默认与 model_path 相同）
            gpu_memory_utilization: GPU 内存使用率
            max_model_len: 最大序列长度
            tensor_parallel_size: Tensor 并行大小
            trust_remote_code: 是否信任远程代码
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path

        try:
            from vllm import LLM
            self.vllm_available = True
        except ImportError:
            logger.warning("vLLM not available. Install with: pip install vllm")
            self.vllm_available = False
            self.llm = None
            return

        logger.info(f"Initializing vLLM with model: {model_path}")
        self.llm = LLM(
            model=self.model_path,
            tokenizer=self.tokenizer_path,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
        )
        logger.info("vLLM model loaded successfully")

    def run_model_multiple_output(
        self,
        model_input: Mapping[str, np.ndarray] | np.ndarray,
        model_name: str = "default",
        model_version: Optional[int] = None,
        model_output_keys: Optional[Set[str]] = None,
        parameters: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, np.ndarray]:
        """
        运行推理（兼容 MedGemma ModelRunner 接口）

        Args:
            model_input: 输入数据（字典或 numpy 数组）
            model_name: 模型名称
            model_version: 模型版本
            model_output_keys: 输出键集合
            parameters: 推理参数

        Returns:
            输出字典
        """
        if not self.vllm_available or self.llm is None:
            raise RuntimeError("vLLM is not available")

        from vllm import SamplingParams

        parameters = parameters or {}
        model_output_keys = model_output_keys or {"text_output"}

        # 提取 prompt
        if isinstance(model_input, dict):
            prompt = model_input.get("prompt", "")
            if isinstance(prompt, np.ndarray):
                prompt = prompt.tobytes().decode('utf-8')
        else:
            # 如果是 numpy 数组，解码为字符串
            prompt = model_input.tobytes().decode('utf-8')

        # 创建采样参数
        sampling_params = SamplingParams(
            max_tokens=parameters.get("max_tokens", 500),
            temperature=parameters.get("temperature", 0.7),
            top_p=parameters.get("top_p", 0.9),
            top_k=parameters.get("top_k", -1),
            n=parameters.get("n", 1),
        )

        # 运行推理
        outputs = self.llm.generate([prompt], sampling_params)

        # 格式化输出
        result = {}
        if "text_output" in model_output_keys:
            result["text_output"] = np.array([
                output.text.encode('utf-8') for output in outputs[0].outputs
            ])
        if "num_input_tokens" in model_output_keys:
            result["num_input_tokens"] = np.array([len(outputs[0].prompt_token_ids)])
        if "num_output_tokens" in model_output_keys:
            result["num_output_tokens"] = np.array([
                len(output.token_ids) for output in outputs[0].outputs
            ])

        return result

    def generate(
        self,
        prompts: List[str],
        max_tokens: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = -1,
        n: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        简化的生成接口

        Args:
            prompts: 输入提示列表
            max_tokens: 最大生成 token 数
            temperature: 采样温度
            top_p: Nucleus 采样参数
            top_k: Top-K 采样参数
            n: 生成数量

        Returns:
            生成结果列表
        """
        if not self.vllm_available or self.llm is None:
            raise RuntimeError("vLLM is not available")

        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            n=n,
        )

        outputs = self.llm.generate(prompts, sampling_params)

        results = []
        for output in outputs:
            result = {
                "prompt": output.prompt,
                "outputs": [
                    {
                        "text": o.text,
                        "token_ids": o.token_ids,
                        "cumulative_logprob": o.cumulative_logprob,
                        "finish_reason": o.finish_reason,
                    }
                    for o in output.outputs
                ],
                "num_input_tokens": len(output.prompt_token_ids),
            }
            results.append(result)

        return results


class TransformersModelRunner:
    """
    基于 Transformers 的备用模型运行器
    当 vLLM 不可用时使用
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        device: str = "cuda",
        torch_dtype: str = "auto",
    ):
        """
        初始化 Transformers 模型运行器

        Args:
            model_path: 模型路径
            tokenizer_path: Tokenizer 路径
            device: 设备（cuda 或 cpu）
            torch_dtype: 数据类型
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.tokenizer_path = tokenizer_path or model_path

        logger.info(f"Loading model with transformers: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype if torch_dtype != "auto" else "auto",
            device_map=device,
        )
        logger.info("Model loaded successfully")

    def generate(
        self,
        prompts: List[str],
        max_tokens: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        n: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        生成文本

        Args:
            prompts: 输入提示列表
            max_tokens: 最大生成 token 数
            temperature: 采样温度
            top_p: Nucleus 采样参数
            top_k: Top-K 采样参数
            n: 生成数量

        Returns:
            生成结果列表
        """
        import torch

        results = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    num_return_sequences=n,
                    do_sample=temperature > 0,
                )

            generated_texts = [
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]

            result = {
                "prompt": prompt,
                "outputs": [
                    {
                        "text": text,
                        "token_ids": None,
                        "cumulative_logprob": None,
                        "finish_reason": "stop",
                    }
                    for text in generated_texts
                ],
                "num_input_tokens": len(inputs["input_ids"][0]),
            }
            results.append(result)

        return results
