import json
from typing import Dict
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from agent.rag.datasource.vdb.milvus.schemas import COLLECTION_TO_INFOS
from agent.llm import ChatModel

SYSTEM_TEMPLATE = """
你是一个智能检索助手，负责根据用户任务从可用的向量数据库集合中精准召回最相关的内容集合。

请仔细分析以下信息：
-用户任务：理解任务的核心目标、涉及的领域、所需的信息类型（如事实、文档、日志、产品信息等）。
-集合详情：每个集合包含名称和内容摘要，摘要描述了该集合所存储的数据类型、主题范围和典型内容。
-你的目标是：基于任务需求，判断哪些集合在语义或内容上与任务高度相关，能够为任务执行提供支持。

要求：
-深入理解任务意图，避免表面关键词匹配。
-结合集合摘要进行语义匹配，判断其内容是否可能包含任务所需信息。
-只输出最相关的一个或多个集合名称；若无相关集合，输出空列表。
-输出必须为 JSON 格式，且仅包含 collection_names 字段，其值为字符串列表。

任务:
{task}

集合详情:
{collection_info}

请严格遵循以下JSON Schema输出：
{format_instructions}
"""


class MatchResult(BaseModel):
    collection_names: list[str] = Field(description="集合的名称")


class CollectionPreprocessor:

    def __init__(self, config: Dict):
        self.llm = ChatModel[config["factory_name"]](**config["model"])

    def preprocess(self, task: str) -> list[str]:
        parser = JsonOutputParser(pydantic_object=MatchResult)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_TEMPLATE),
            ]
        ).partial(format_instructions=parser.get_format_instructions())
        preprocess_chain = prompt | self.llm | parser
        collection_info = [
            f"集合名称：{k}, 集合存储内容摘要：{v}"
            for k, v in COLLECTION_TO_INFOS.items()
        ]
        response = preprocess_chain.invoke(
            {"task": task, "collection_info": collection_info}
        )
        result = MatchResult.model_validate_json(json.dumps(response))
        print(f"collection_names: {result.collection_names}")
        return result.collection_names
