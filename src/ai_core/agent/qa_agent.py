import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "unit"))
from yaml import safe_load
from typing import Optional, Any, Iterator, Dict

from langchain_core.runnables.utils import Input, Output
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, END
from langgraph.typing import StateLike

from ai_core.agent.unit.qa_generator import QaGenerator


class QaAgent(Runnable):

    def __init__(
        self, config: Dict, state_schema: type(StateLike), mem: InMemorySaver = None
    ):
        task_runner = QaGenerator(config)

        workflow = StateGraph(state_schema)
        workflow.add_node("execute", task_runner.node)

        workflow.set_entry_point("execute")
        workflow.add_edge("execute", END)

        memery = mem or InMemorySaver()
        self.workflow = workflow.compile(checkpointer=memery)

    def invoke(
        self,
        input: Input,  # noqa: A002
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Output:
        return self.workflow.invoke(input, config)

    def stream(
        self,
        input: Input,  # noqa: A002
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[Output]:
        for s in self.workflow.stream(input=input, config=config):
            print(s)
            print("-" * 50)
        final_state = self.workflow.get_state(config=config)
        print("✅ 最终答案：")
        print(final_state.values.get("task_result", "无"))


if __name__ == "__main__":
    from ai_core.agent.unit.agent_state import AgentState

    config = {"configurable": {"thread_id": "thread_1"}, "recursion_limit": 50}
    with open("../config/onescience_agent.yml", "r") as f:
        agent_config = safe_load(f)
        agent_config["user"] = "abcd"

    inputs = {"messages": [("user", "四川的省会城市是哪个")]}
    react = QaAgent(agent_config, AgentState)
    react.stream(inputs, config)
