#!/user/bin/env
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "unit"))

from yaml import safe_load
from typing import Dict, Optional, Any, Iterator
from langgraph.typing import StateLike
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.utils import Input, Output

from ai_core.agent.unit.cot_runner import CotRunner
from ai_core.agent.unit.tool_runner import ToolRunner
from ai_core.until.untils import pretty_print


class CotAgent(Runnable):
    def __init__(
        self, config: Dict, state_schema: type(StateLike), mem: InMemorySaver = None
    ):
        workflow = StateGraph(state_schema)
        call_model = CotRunner(config)
        tool_node = ToolRunner(config)

        # Define the two nodes we will cycle between
        workflow.add_node("agent", call_model.node)
        workflow.add_node("tools", tool_node.node)

        # Set the entrypoint as `agent`
        workflow.set_entry_point("agent")

        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            self.should_continue,
            {
                "continue": "tools",
                "end": END,
            },
        )

        # Add edge from tools back to agent
        workflow.add_edge("tools", "agent")

        # Compile the graph
        self.workflow = workflow.compile(checkpointer=mem or InMemorySaver())

    def invoke(
        self,
        input: Input,  # noqa: A002
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Output:
        return self.workflow.invoke(input, config)

    @staticmethod
    def should_continue(state: type(StateLike)):
        """
        Determine if we should continue running the graph or finish.
        """

        messages = state["messages"]
        last_message = messages[-1]
        print(f"should_continue {last_message}")
        # If there is no tool call, then we finish
        if not hasattr(last_message, "tool_calls") or not len(last_message.tool_calls):
            return "end"
        # Otherwise if there is, we continue
        else:
            return "continue"

    def stream(
        self,
        input: Input,  # noqa: A002
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[Output]:

        log = []
        for s in self.workflow.stream(input, stream_mode="values", config=config):
            message = s["messages"][-1]
            out = pretty_print(message)
            log.append(out)
            print(s)
            print("-" * 50)

        print("\t".join(log))
        final_state = self.workflow.get_state(config=config)
        print("✅ 最终答案：")
        print(final_state.values.get("task_result", "无"))

        return log


if __name__ == "__main__":
    from ai_core.agent.unit.agent_state import AgentState

    config = {"configurable": {"thread_id": "thread_1"}, "recursion_limit": 50}
    with open("../config/onescience_agent.yml", "r") as f:
        agent_config = safe_load(f)
        agent_config["user"] = "abcd"

    inputs = {"messages": [("user", "四川的省会城市是哪个")]}
    react = CotAgent(agent_config, AgentState)
    react.stream(inputs, config)
