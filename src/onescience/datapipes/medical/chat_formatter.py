# 医学数据加载器 - Chat 格式化器
# 将医学数据转换为 OpenAI Chat Completion 格式

from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ChatFormatter:
    """
    OpenAI Chat Completion 格式化器
    """

    def __init__(self):
        """初始化格式化器"""
        pass

    def format_messages(
        self,
        user_content: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        """
        格式化消息为 Chat Completion 格式

        Args:
            user_content: 用户输入内容
            system_prompt: 系统提示（可选）
            history: 历史对话（可选）

        Returns:
            格式化的消息列表
        """
        messages = []

        # 添加系统提示
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # 添加历史对话
        if history:
            messages.extend(history)

        # 添加当前用户输入
        messages.append({"role": "user", "content": user_content})

        return messages

    def format_medical_query(
        self,
        question: str,
        patient_info: Optional[Dict[str, Any]] = None,
        clinical_context: Optional[str] = None,
        image_description: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        格式化医学查询

        Args:
            question: 医学问题
            patient_info: 患者信息（年龄、性别等）
            clinical_context: 临床背景
            image_description: 图像描述

        Returns:
            格式化的消息列表
        """
        # 构建上下文
        context_parts = []

        if patient_info:
            patient_str = ", ".join(
                f"{k}: {v}" for k, v in patient_info.items()
            )
            context_parts.append(f"Patient Information: {patient_str}")

        if clinical_context:
            context_parts.append(f"Clinical Context: {clinical_context}")

        if image_description:
            context_parts.append(f"Imaging: {image_description}")

        # 组合完整内容
        if context_parts:
            full_content = "\n\n".join(context_parts) + f"\n\nQuestion: {question}"
        else:
            full_content = question

        return [
            {
                "role": "system",
                "content": "You are an expert medical AI assistant. Provide accurate and helpful medical information."
            },
            {
                "role": "user",
                "content": full_content
            }
        ]

    def format_radiology_report_request(
        self,
        modality: str,
        clinical_indication: str,
        findings: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        格式化放射学报告请求

        Args:
            modality: 影像模态（CT、MRI、X-ray 等）
            clinical_indication: 临床适应症
            findings: 已有发现（可选）

        Returns:
            格式化的消息列表
        """
        content = f"Modality: {modality}\nClinical Indication: {clinical_indication}"

        if findings:
            content += f"\n\nFindings: {findings}\n\nPlease provide a detailed impression and recommendation."
        else:
            content += "\n\nPlease analyze the image and provide findings, impression, and recommendations."

        return [
            {
                "role": "system",
                "content": "You are an expert radiologist. Generate detailed and accurate radiology reports."
            },
            {
                "role": "user",
                "content": content
            }
        ]

    def format_diagnosis_request(
        self,
        symptoms: List[str],
        patient_history: Optional[str] = None,
        lab_results: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, str]]:
        """
        格式化诊断请求

        Args:
            symptoms: 症状列表
            patient_history: 病史
            lab_results: 实验室检查结果

        Returns:
            格式化的消息列表
        """
        content_parts = [
            f"Symptoms: {', '.join(symptoms)}"
        ]

        if patient_history:
            content_parts.append(f"Patient History: {patient_history}")

        if lab_results:
            lab_str = "\n".join(
                f"  - {k}: {v}" for k, v in lab_results.items()
            )
            content_parts.append(f"Lab Results:\n{lab_str}")

        content_parts.append(
            "\nPlease provide a differential diagnosis and recommended next steps."
        )

        return [
            {
                "role": "system",
                "content": "You are an experienced clinician. Provide thoughtful differential diagnoses based on the available information."
            },
            {
                "role": "user",
                "content": "\n\n".join(content_parts)
            }
        ]

    def parse_response(self, response: Dict[str, Any]) -> str:
        """
        解析 Chat Completion 响应

        Args:
            response: OpenAI 格式的响应

        Returns:
            提取的文本内容
        """
        if "choices" in response and len(response["choices"]) > 0:
            return response["choices"][0]["message"]["content"]
        return ""

    def extract_all_responses(self, response: Dict[str, Any]) -> List[str]:
        """
        提取所有响应（n > 1 时）

        Args:
            response: OpenAI 格式的响应

        Returns:
            所有响应文本列表
        """
        responses = []
        if "choices" in response:
            for choice in response["choices"]:
                responses.append(choice["message"]["content"])
        return responses
