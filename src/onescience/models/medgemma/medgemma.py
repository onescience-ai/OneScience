# MedGemma 主模型类
# 继承 OneScience Module 基类，集成到 OneScience 框架

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from onescience.modules import Module
from onescience.models.medgemma.model_runner import VLLMModelRunner, TransformersModelRunner
from onescience.models.medgemma.predictor_wrapper import MedGemmaPredictor

logger = logging.getLogger(__name__)


class MedGemma(Module):
    """
    MedGemma: 医学大语言模型

    支持:
    - 4B 多模态模型（文本 + 医学图像）
    - 27B 文本模型
    - DICOM/CT/CXR/WSI 图像输入
    - OpenAI Chat Completion API 格式

    继承 OneScience Module 基类，提供统一接口
    """

    def __init__(self, configs: Any) -> None:
        """
        初始化 MedGemma 模型

        Args:
            configs: 配置对象（ConfigDict）
        """
        super(MedGemma, self).__init__()
        self.configs = configs

        # 模型变体（4B 或 27B）
        self.model_variant = configs.model.variant
        self.is_multimodal = configs.model.is_multimodal

        logger.info(f"Initializing MedGemma {self.model_variant} model")
        logger.info(f"Multimodal: {self.is_multimodal}")

        # 初始化模型运行器
        self._init_model_runner()

        # 初始化推理包装器
        self.predictor = MedGemmaPredictor(
            model_runner=self.model_runner,
            configs=configs,
        )

        logger.info("MedGemma model initialized successfully")

    def _init_model_runner(self):
        """初始化模型运行器（vLLM 或 Transformers）"""
        if self.configs.inference.use_vllm:
            try:
                self.model_runner = VLLMModelRunner(
                    model_path=self.configs.model.model_path,
                    tokenizer_path=self.configs.model.tokenizer_path,
                    gpu_memory_utilization=self.configs.inference.gpu_memory_utilization,
                    max_model_len=self.configs.inference.max_model_len,
                    tensor_parallel_size=self.configs.inference.tensor_parallel_size,
                )
                logger.info("Using vLLM model runner")
            except Exception as e:
                logger.warning(f"vLLM initialization failed: {e}")
                logger.info("Falling back to Transformers model runner")
                self._init_transformers_runner()
        else:
            self._init_transformers_runner()

    def _init_transformers_runner(self):
        """初始化 Transformers 运行器"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_runner = TransformersModelRunner(
            model_path=self.configs.model.model_path,
            tokenizer_path=self.configs.model.tokenizer_path,
            device=device,
        )
        logger.info("Using Transformers model runner")

    def forward(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        n: int = 1,
    ) -> Dict[str, Any]:
        """
        前向传播（使用 OpenAI Chat Completion 格式）

        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            max_tokens: 最大生成 token 数
            temperature: 采样温度（0-2）
            top_p: Nucleus 采样参数（0-1）
            n: 生成数量

        Returns:
            OpenAI 格式的响应字典
        """
        if max_tokens is None:
            max_tokens = self.configs.inference.default_max_tokens

        return self.predictor.predict(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n,
        )

    @torch.no_grad()
    def inference(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        推理方法（兼容 BiologyInferenceRunner）

        Args:
            data: 输入数据字典，可包含:
                - messages: 消息列表
                - instances: 实例列表（用于批处理）
                - parameters: 推理参数

        Returns:
            预测结果字典
        """
        # 转换数据格式
        if "instances" in data:
            messages = self._convert_instances_to_messages(data["instances"])
        elif "messages" in data:
            messages = data["messages"]
        else:
            raise ValueError("Input data must contain 'messages' or 'instances'")

        # 提取推理参数
        parameters = data.get("parameters", {})
        max_tokens = parameters.get("max_tokens", self.configs.inference.default_max_tokens)
        temperature = parameters.get("temperature", self.configs.inference.temperature)
        top_p = parameters.get("top_p", self.configs.inference.top_p)
        n = parameters.get("n", 1)

        # 运行推理
        return self.forward(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n,
        )

    def _convert_instances_to_messages(
        self,
        instances: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        将实例列表转换为消息格式

        Args:
            instances: 实例列表

        Returns:
            消息列表
        """
        messages = []
        for instance in instances:
            if "role" in instance and "content" in instance:
                messages.append(instance)
            elif "text" in instance:
                messages.append({"role": "user", "content": instance["text"]})
            elif "question" in instance:
                messages.append({"role": "user", "content": instance["question"]})
            else:
                logger.warning(f"Unknown instance format: {instance}")

        return messages

    def predict_text(
        self,
        text: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> str:
        """
        简化的文本预测接口

        Args:
            text: 输入文本
            max_tokens: 最大生成 token 数
            temperature: 采样温度

        Returns:
            生成的文本
        """
        messages = [{"role": "user", "content": text}]
        response = self.forward(messages, max_tokens, temperature)

        if response["choices"]:
            return response["choices"][0]["message"]["content"]
        return ""

    def predict_multimodal(
        self,
        text: str,
        images: List[Any],
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> str:
        """
        多模态预测接口（文本 + 图像）

        Args:
            text: 输入文本
            images: 图像列表
            max_tokens: 最大生成 token 数
            temperature: 采样温度

        Returns:
            生成的文本
        """
        if not self.is_multimodal:
            logger.warning("Model is not multimodal, ignoring images")
            return self.predict_text(text, max_tokens, temperature)

        # TODO: 实现多模态推理
        # 需要集成图像编码器和多模态提示格式
        messages = [{"role": "user", "content": text}]
        response = self.predictor.predict_with_images(
            messages=messages,
            images=images,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        if response["choices"]:
            return response["choices"][0]["message"]["content"]
        return ""

    def __repr__(self) -> str:
        return f"MedGemma(variant={self.model_variant}, multimodal={self.is_multimodal})"
