#!/user/bin/env
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.join(os.path.dirname(__file__), "unit"))

import re
import argparse
from yaml import safe_load
from typing import Dict, Optional, Any, Iterator
from langgraph.typing import StateLike
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.utils import Input, Output
from langchain_core.messages import AIMessage

from agent.agent.unit.tool_runner import ToolRunner
from agent.agent.unit.generate_runner import GenerateRunner, ParseRunner
from agent.untils.until import pretty_print


class ReactAgent(Runnable):

    def __init__(self,
            agent_config: Dict,
            rag_config: Optional[Dict],
            state_schema: type(StateLike),
            mem: InMemorySaver = None,
            reflect_times:int = 0,
    ):
        workflow = StateGraph(state_schema)
        tool = ToolRunner(agent_config, rag_config)
        generator = GenerateRunner(agent_config, rag_config)
        parser = ParseRunner(agent_config, rag_config)

        # Define the two nodes we will cycle between
        workflow.add_node("tool", tool.node)
        workflow.add_node("generate", generator.node)
        workflow.add_node("parse", parser.node)

        # Set the entrypoint as `generate`
        workflow.set_entry_point("generate")
        workflow.add_edge("generate", "parse")

        if reflect_times > 0:
            workflow.add_conditional_edges(
                "parse",
                self.routing_function,
                path_map={
                    "generate": "generate",
                    "tool": "tool",
                    "end": "reflect",
                },
            )
            workflow.add_conditional_edges(
                "reflect",
                self.routing_function_reflect,
                path_map={"generate": "generate", "end": END},
            )
        else:
            workflow.add_conditional_edges(
                "parse",
                self.routing_function,
                path_map={
                    "generate": "generate",
                    "tool": "tool",
                    "end": END,
                },
            )
        workflow.add_edge("tool", "generate")

        # Compile the graph
        self.workflow = workflow.compile(checkpointer=mem or InMemorySaver())
        if reflect_times > 0:
            self.cur_reflect_times = 0
            self.reflect_times = reflect_times

    def invoke(
        self,
        input: Input,  # noqa: A002
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Output:
        return self.workflow.invoke(input, config)

    @staticmethod
    def routing_function(state: type(StateLike)) -> str:
        msg = state["messages"][-1]
        print(f"last_msg：：：{msg}")
        if msg.content == "由于重复出现解析错误，执行已终止。请检查您的输入并重试。":
            # If we've already tried to correct the model twice, just end the conversation
            print("Detected repeated parsing errors, ending conversation")
            return "end"
        elif (
            msg.content
            == "每个回复必须包含思考过程，后跟 <think> 或 <solution> 标签。但当前回复中没有这些标签。请遵循指令，修正并重新生成回复。"
        ):
            return "generate"
        elif isinstance(msg, AIMessage) and len(msg.tool_calls):
            return "tool"
        else:
            msg = msg.content
            if "<solution>" in msg and "</solution>" not in msg:
                msg += "</solution>"

            answer_match = re.search(r"<solution>(.*?)</solution>", msg, re.DOTALL)
            if answer_match:
                return "end"
            else:
                return "generate"

    def routing_function_reflect(self, state: type(StateLike)) -> str:
        if self.cur_reflect_times < self.reflect_times:
            self.cur_reflect_times += 1
            print(f"reflect times: {self.cur_reflect_times}")
            return "generate"
        else:
            return "end"

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
    parser.add_argument('--reflect_times', type=int, default=0, description="reflect次数")
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
    react = ReactAgent(agent_config=agent_config,
                       rag_config=rag_config,
                       state_schema=AgentState,
                       reflect_times=args.reflect_times)
    react.stream(inputs, config)
