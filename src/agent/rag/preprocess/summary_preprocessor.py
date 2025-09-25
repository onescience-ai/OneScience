from typing import Dict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from agent.llm import ChatModel

SYSTEM_TEMPLATE = """
请对以下内容进行精准摘要，提取其核心信息，要求：
-紧扣原文主题，准确保留关键事实、数据和结论
-去除冗余细节和重复信息，突出主要观点
-不添加任何解释、评价或个人观点
-语言简洁流畅，逻辑清晰连贯
-严格控制在500字以内
-仅输出摘要内容，不加标题、说明或提示语

直接输出符合上述要求的总结文本
"""


class SummaryPreprocessor:

    def __init__(self, config: Dict):
        self.llm = ChatModel[config["factory_name"]](**config["model"])

    def preprocess(self, doc: Document) -> Document:
        prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_TEMPLATE), ("user", "内容：\n{content}\n")]
        )
        preprocess_chain = prompt | self.llm
        response = preprocess_chain.invoke(
            {"content": "\n".join([doc.page_content, doc.metadata["title"]])}
        )

        doc.metadata["para_summary"] = response.content

        return doc
