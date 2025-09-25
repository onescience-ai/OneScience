#!/user/bin/env
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.join(os.path.dirname(__file__), "unit"))

from yaml import safe_load
from typing import Dict, Optional, Any, Iterator
from langgraph.typing import StateLike
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.utils import Input, Output

from agent.agent.unit.cot_runner import CotRunner
from agent.agent.unit.tool_runner import ToolRunner
from agent.untils.until import pretty_print


class CotAgent(Runnable):
    def __init__(self,
            agent_config: Dict,
            rag_config:Optional[Dict],
            state_schema: type(StateLike), mem: InMemorySaver = None
    ):
        workflow = StateGraph(state_schema)
        call_model = CotRunner(agent_config, rag_config)
        tool_node = ToolRunner(agent_config, rag_config)

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

        logs = []
        for s in self.workflow.stream(input, stream_mode="values", config=config):
            message = s["messages"][-1]
            out = pretty_print(message)
            logs.append(out)
            print(s)
            print("-" * 50)

        print("\t".join(logs))
        final_state = self.workflow.get_state(config=config)
        print("✅ 最终答案：")
        print(final_state.values.get("task_result", "无"))

        return logs


if __name__ == "__main__":
    import uuid
    from agent.untils.until import setup_global_logger
    from agent.agent.unit.agent_state import AgentState
    from agent.untils.until import gen_job_log_name

    parser = argparse.ArgumentParser()
    parser.add_argument('--job_name', type=str, required=True, description="任务名称")
    parser.add_argument('--human_message', type=str, required=True, description="用户任务输入")
    parser.add_argument('--agent_config_path', type=str, required=True, description="Agent配置文件")
    parser.add_argument('--rag_config_path', type=str, default=None, description="RAG配置文件,若不使用RAG可为空")
    parser.add_argument('--recursion_limit', type=int, default=50, description="langgraph的recursion_limit参数")
    args = parser.parse_args()

    log_name = gen_job_log_name(args.job_name, "../logs")
    setup_global_logger(f"../logs/{log_name}")

    user = uuid.uuid4()
    config = {"configurable": {"thread_id": f"thread_{user}"}, "recursion_limit": args.recursion_limit}
    with open(parser.agent_config_path, "r") as f:
        agent_config = safe_load(f)
        agent_config["user"] = user

    rag_config = None
    with open(parser.rag_config_path, "r") as f:
        rag_config = safe_load(f)
        rag_config["user"] = user

    inputs = {"messages": [("user", args.human_message)]}
    cot_agent = CotAgent(agent_config=agent_config,
                       rag_config=rag_config,
                       state_schema=AgentState)
    cot_agent.stream(inputs, config)

