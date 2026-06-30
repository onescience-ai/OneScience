# MedGemma 数据适配器
# 将医学数据格式转换为 MedGemma 输入格式

from typing import Any, Dict, List, Optional
import numpy as np
import logging

from onescience.datapipes.biology.adapters.base_adapter import BaseAdapter, FeatureDict
from onescience.datapipes.core.config import DatasetConfig

logger = logging.getLogger(__name__)


class MedGemmaInferAdapter(BaseAdapter):
    """
    MedGemma 推理适配器

    支持:
    - 文本输入（临床笔记、问题）
    - 多模态输入（文本 + 医学图像）
    - DICOM 图像加载
    - OpenAI Chat Completion 格式
    """

    def __init__(self, config: Optional[DatasetConfig] = None):
        """
        初始化适配器

        Args:
            config: 数据集配置（可选）
        """
        # 如果没有提供配置，创建一个默认配置
        if config is None:
            from ml_collections import ConfigDict
            config = ConfigDict({
                'data': ConfigDict({
                    'extra': {}
                })
            })

        super().__init__(config)

    def adapt_features(self, common_features: FeatureDict) -> FeatureDict:
        """
        将通用特征转换为 MedGemma 格式

        Args:
            common_features: 通用特征字典，可包含:
                - text: str - 临床文本或问题
                - images: List[Dict] - 图像规格
                - messages: List[Dict] - 聊天消息
                - question: str - 医学问题
                - context: str - 上下文信息

        Returns:
            MedGemma 格式的特征字典
        """
        medgemma_features = {}

        # 处理文本输入
        if "text" in common_features:
            medgemma_features["messages"] = [
                {"role": "user", "content": common_features["text"]}
            ]

        # 处理问题格式
        elif "question" in common_features:
            content = common_features["question"]
            if "context" in common_features:
                content = f"Context: {common_features['context']}\n\nQuestion: {content}"
            medgemma_features["messages"] = [
                {"role": "user", "content": content}
            ]

        # 处理聊天消息
        elif "messages" in common_features:
            medgemma_features["messages"] = common_features["messages"]

        else:
            # 默认空消息
            medgemma_features["messages"] = []

        # 处理图像输入
        if "images" in common_features:
            medgemma_features["images"] = common_features["images"]

        # 添加推理参数
        medgemma_features["parameters"] = {
            "max_tokens": common_features.get("max_tokens", 500),
            "temperature": common_features.get("temperature", 0.7),
            "top_p": common_features.get("top_p", 0.9),
        }

        return medgemma_features

    def process_sample(self, sample: Dict[str, Any]) -> FeatureDict:
        """
        处理单个样本

        Args:
            sample: 原始样本数据

        Returns:
            处理后的特征字典
        """
        # 直接适配特征
        return self.adapt_features(sample)

    def process_medical_sample(
        self,
        clinical_note: str,
        images: Optional[List[Dict[str, Any]]] = None,
        task: str = "diagnosis",
        system_prompt: Optional[str] = None,
    ) -> FeatureDict:
        """
        处理医学样本（文本 + 图像）

        Args:
            clinical_note: 临床笔记或问题
            images: 图像规格列表（DICOM 路径、URL 等）
            task: 任务类型（diagnosis、report_generation 等）
            system_prompt: 系统提示（可选）

        Returns:
            MedGemma 格式的特征字典
        """
        messages = []

        # 添加系统提示
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        elif task:
            # 基于任务类型的默认系统提示
            task_prompts = {
                "diagnosis": "You are an expert medical AI assistant. Analyze the provided clinical information and images to assist with diagnosis.",
                "report_generation": "You are an expert radiologist. Generate a detailed medical report based on the provided images and clinical context.",
                "question_answering": "You are a knowledgeable medical AI. Answer the following medical question accurately and concisely.",
            }
            if task in task_prompts:
                messages.append({"role": "system", "content": task_prompts[task]})

        # 添加用户消息
        messages.append({"role": "user", "content": clinical_note})

        features = {
            "messages": messages,
            "task": task,
        }

        if images:
            features["images"] = images

        return features

    def process_dicom_sample(
        self,
        dicom_path: str,
        question: str,
        modality: str = "CT",
    ) -> FeatureDict:
        """
        处理 DICOM 图像样本

        Args:
            dicom_path: DICOM 文件路径
            question: 关于图像的问题
            modality: 影像模态（CT、MRI、CXR 等）

        Returns:
            处理后的特征字典
        """
        features = {
            "messages": [
                {
                    "role": "user",
                    "content": f"[{modality} Image]\n\n{question}"
                }
            ],
            "images": [
                {
                    "type": "dicom",
                    "path": dicom_path,
                    "modality": modality,
                }
            ],
        }

        return features

    def process_json_sample(
        self,
        json_data: Dict[str, Any]
    ) -> FeatureDict:
        """
        处理 JSON 格式的样本

        Args:
            json_data: JSON 数据，可包含:
                - messages: 聊天消息
                - text: 文本内容
                - question: 问题
                - images: 图像列表
                - context: 上下文

        Returns:
            处理后的特征字典
        """
        # 使用 JSON 解析器
        parsed_data = self.json_parser.parse(json_data)

        # 适配为 MedGemma 格式
        return self.adapt_features(parsed_data)

    def batch_process(
        self,
        samples: List[Dict[str, Any]]
    ) -> List[FeatureDict]:
        """
        批量处理样本

        Args:
            samples: 样本列表

        Returns:
            处理后的特征列表
        """
        return [self.process_sample(sample) for sample in samples]
