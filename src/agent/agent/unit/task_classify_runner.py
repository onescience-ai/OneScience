import json
import operator
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.typing import StateLike
from typing import Dict, Optional
from agent.agent.unit.runner import Runner


class TaskTypeResult(BaseModel):
    task_type: str = Field(description="任务的类型")


class TaskClasser(Runner):
    name = "task_classer"
    system_prompt = """
请根据用户请求的内容和可用工具列表，判断该任务属于以下哪一类：

1. **简单问答**（simple_qa）：仅需基于已有知识即可回答，无需调用外部工具或执行计算。
2. **需要调用工具**（needs_tool）：需要使用指定工具（如搜索、计算、API 调用等）获取信息或完成操作。
3. **复杂任务**（complex_task）：涉及多个步骤或子任务，需拆解并可能结合工具调用才能完成。

可用工具:
{tools}

上下文:
{context}

请仔细分析用户请求是否：
- 可直接回答 → simple_qa
- 需要调用一个或多个工具 → needs_tool
- 包含多个目标或依赖顺序执行的子任务 → complex_task

输出必须为JSON格式，且仅包含`task_type`字段，取值为以下三者之一：
"simple_qa", "needs_tool", "complex_task"

请确保输出严格符合以下 JSON Schema：
{format_instructions}
"""

    def __init__(self, agent_config: Dict, rag_config: Optional[Dict] = None):
        super().__init__(agent_config, rag_config)

    def node(self, state: StateLike) -> Dict:
        context = []
        if self.rag:
            context = self.rag.retrieve(user=self.user,
                                        query=state["messages"][-1],
                                        retrieval_method="hybrid_search")
        context = "".join(context)

        parser = JsonOutputParser(pydantic_object=TaskTypeResult)

        messages = [SystemMessage(self.system_prompt)] + state["messages"]
        runnable = ChatPromptTemplate.from_messages(messages) | self.llm | parser

        result = runnable.invoke(
            {
                "tools": self.tools,
                "context": context,
                "format_instructions": parser.get_format_instructions(),
            }
        )

        result = TaskTypeResult.model_validate_json(json.dumps(result))
        return {"messages": [AIMessage(content=result.task_type)]}
