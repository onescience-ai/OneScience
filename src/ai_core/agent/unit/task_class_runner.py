import json
import operator
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.typing import StateLike
from typing import Dict
from ai_core.agent.unit.runner import Runner


class TaskTypeResult(BaseModel):
    task_type: str = Field(description="任务的类型")


class TaskClasser(Runner):
    name = "task_classer"
    system_prompt = """
请根据提供的工具判断用户请求的任务属于下面哪一类：
1. 简单问答（基于知识即可回答）
2. 需要调用工具（如搜索、计算、API）
3. 复杂任务（需拆解为多个子任务）

【工具】
{tools}

以JSON格式输出，只有task_type字段，输出值simple_qa | needs_tool | complex_task，
JSON格式必须符合以下schema:
{format_instructions}
"""

    def __init__(self, config: Dict):
        super().__init__(config)

    def node(self, state: StateLike) -> Dict:
        # context = self.rag.retrieve(user=self.user,
        #                             query=task,
        #                             retrieval_method="hybrid_search")

        parser = JsonOutputParser(pydantic_object=TaskTypeResult)

        messages = [SystemMessage(self.system_prompt)] + state["messages"]
        runnable = ChatPromptTemplate.from_messages(messages) | self.llm | parser

        result = runnable.invoke(
            {
                "tools": self.tools,
                "format_instructions": parser.get_format_instructions(),
            }
        )

        result = TaskTypeResult.model_validate_json(json.dumps(result))
        return {"messages": [AIMessage(content=result.task_type)]}
