import json
from typing import Dict
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from ai_core.rag.datasource.vdb.milvus.schemas import collection_to_infos
from ai_core.llm import ChatModel

SYSTEM_TEMPLATE = """
用户的任务执行需要从给定的集合中召回相关的内容，现给定用户要执行的任务和现有集合存储的内容请根据执行的任务匹配出检索集合
要求：
- 仔细理解任务
- 仔细理解集合，根据集合存储的信息摘要知道其都存储了哪些内容
- 直接输出匹配集合名称

【任务】
{task}

【集合详情】
{collection_info}

以JSON格式输出，只有collection_names字段，是一个list结构为匹配出的集合名称，JSON格式必须符合以下schema:
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
            for k, v in collection_to_infos.items()
        ]
        response = preprocess_chain.invoke(
            {"task": task, "collection_info": collection_info}
        )
        result = MatchResult.model_validate_json(json.dumps(response))
        print(f"collection_names: {result.collection_names}")
        return result.collection_names
