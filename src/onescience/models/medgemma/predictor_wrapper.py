# MedGemma 推理包装器
# 包装 MedGemma 原始 predictor.py 的推理逻辑

import logging
import os
import sys
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MedGemmaPredictor:
    """
    MedGemma 推理包装器
    包装原始 MedGemma predictor 逻辑，提供 OneScience 兼容接口
    """

    def __init__(self, model_runner: Any, configs: Any):
        """
        初始化推理包装器

        Args:
            model_runner: 模型运行器（VLLMModelRunner 或 TransformersModelRunner）
            configs: 配置对象
        """
        self.model_runner = model_runner
        self.configs = configs

        # 尝试导入原始 MedGemma 组件（如果可用）
        self._init_medgemma_components()

    def _init_medgemma_components(self):
        """初始化 MedGemma 原始组件"""
        try:
            # 添加 MedGemma 原始代码路径到 sys.path
            medgemma_base = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "..", "medgemma", "python")
            )

            if os.path.exists(medgemma_base) and medgemma_base not in sys.path:
                sys.path.insert(0, medgemma_base)
                logger.info(f"Added MedGemma path: {medgemma_base}")

            # 尝试导入 MedGemma predictor 组件
            try:
                from serving import predictor
                self.has_original_predictor = True
                logger.info("Successfully imported original MedGemma predictor")
            except ImportError as e:
                logger.warning(f"Could not import original MedGemma predictor: {e}")
                self.has_original_predictor = False

        except Exception as e:
            logger.warning(f"Error initializing MedGemma components: {e}")
            self.has_original_predictor = False

    def predict(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.9,
        n: int = 1,
    ) -> Dict[str, Any]:
        """
        运行推理

        Args:
            messages: OpenAI Chat Completion 格式的消息列表
            max_tokens: 最大生成 token 数
            temperature: 采样温度
            top_p: Nucleus 采样参数
            n: 生成数量

        Returns:
            OpenAI 兼容格式的响应
        """
        # 转换消息为 prompt
        prompt = self._messages_to_prompt(messages)

        # 运行模型推理
        results = self.model_runner.generate(
            prompts=[prompt],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n,
        )

        # 格式化响应为 OpenAI 格式
        return self._format_openai_response(results[0], messages)

    def _messages_to_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """
        将 OpenAI 消息格式转换为 prompt

        Args:
            messages: 消息列表

        Returns:
            格式化的 prompt 字符串
        """
        prompt_parts = []

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            # 处理不同角色的消息
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            else:
                prompt_parts.append(f"{role}: {content}")

        # 添加 Assistant 前缀以开始生成
        prompt_parts.append("Assistant:")

        return "\n".join(prompt_parts)

    def _format_openai_response(
        self,
        result: Dict[str, Any],
        messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        将模型输出格式化为 OpenAI Chat Completion 格式

        Args:
            result: 模型生成结果
            messages: 原始消息

        Returns:
            OpenAI 格式的响应
        """
        import time
        import uuid

        choices = []
        for idx, output in enumerate(result["outputs"]):
            choice = {
                "index": idx,
                "message": {
                    "role": "assistant",
                    "content": output["text"].replace(result["prompt"], "").strip(),
                },
                "finish_reason": output.get("finish_reason", "stop"),
            }
            choices.append(choice)

        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.configs.model.variant,
            "choices": choices,
            "usage": {
                "prompt_tokens": result.get("num_input_tokens", 0),
                "completion_tokens": sum(
                    len(output.get("token_ids", [])) if output.get("token_ids") else 0
                    for output in result["outputs"]
                ),
                "total_tokens": result.get("num_input_tokens", 0) + sum(
                    len(output.get("token_ids", [])) if output.get("token_ids") else 0
                    for output in result["outputs"]
                ),
            },
        }

        return response

    def predict_with_images(
        self,
        messages: List[Dict[str, Any]],
        images: List[Any],
        max_tokens: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Dict[str, Any]:
        """
        多模态推理（文本 + 图像）

        Args:
            messages: 消息列表
            images: 图像列表
            max_tokens: 最大生成 token 数
            temperature: 采样温度
            top_p: Nucleus 采样参数

        Returns:
            响应字典
        """
        # TODO: 实现多模态推理
        # 这需要集成 MedGemma 的图像处理逻辑
        logger.warning("Multimodal inference not yet fully implemented")

        # 暂时只处理文本
        return self.predict(messages, max_tokens, temperature, top_p)
