from typing import Dict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from ai_core.llm import ChatModel

SYSTEM_TEMPLATE = """
请对用户输入内容进行简洁、准确的总结，突出主要观点和关键信息
要求：
- 保留主题信息
- 保留核心事实和结论
- 避免添加个人观点
- 语言通顺、逻辑清晰
- 总结内容不超过500字
- 直接输出总结内容
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
