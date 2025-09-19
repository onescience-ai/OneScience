from langchain_core.messages import SystemMessage
from langgraph.typing import StateLike
from typing import List, Annotated, Dict

from ai_core.agent.unit.runner import Runner


class QaGenerator(Runner):
    name = "qa_generator"
    system_prompt = """
请结合上下文信息回答用户的查询

【上下文】
{context}
"""

    def __init__(self, config: Dict):
        super().__init__(config)

    def node(self, state: StateLike) -> Dict:
        query = state["messages"][-1].content
        context = self.rag.retrieve(
            user=self.user, query=query, retrieval_method="hybrid_search"
        )
        messages = [SystemMessage(self.system_prompt.format(context=context))] + state[
            "messages"
        ]
        response = self.llm.invoke(messages)
        return {"messages": response}
