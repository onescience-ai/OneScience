#!/user/bin/env
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.join(os.path.dirname(__file__), "unit"))

import re
import json
from yaml import safe_load
from typing import Dict, Optional, Any, Iterator
from langgraph.typing import StateLike
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.utils import Input, Output
from langchain_core.messages import AIMessage

from ai_core.agent.unit.tool_runner import ToolRunner
from ai_core.agent.unit.generate_runner import GenerateRunner, ParseRunner


class ReactAgent(Runnable):

    def __init__(
        self, config: Dict, state_schema: type(StateLike), mem: InMemorySaver = None
    ):
        workflow = StateGraph(state_schema)
        tool = ToolRunner(config)
        # reflect = ReflectRunner(config)
        generator = GenerateRunner(config)
        parser = ParseRunner(config)

        # Define the two nodes we will cycle between
        workflow.add_node("tool", tool.node)
        # workflow.add_node("reflect", reflect.node)
        workflow.add_node("generate", generator.node)
        workflow.add_node("parse", parser.node)

        # Set the entrypoint as `generate`
        workflow.set_entry_point("generate")
        workflow.add_edge("generate", "parse")

        if "agent" in config and "reflect_times" in config["agent"]:
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
        if "agent" in config and "reflect_times" in config["agent"]:
            self.cur_reflect_times = 0
            self.reflect_times = config["agent"]["reflect_times"]

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

        log = []
        for s in self.workflow.stream(input, stream_mode="values", config=config):
            message = s["messages"][-1]
            # out = pretty_print(message)
            # log.append(out)
            # print(s)
            # print("-" * 50)

        # print("\t".join(log))
        # final_state = self.workflow.get_state(config=config)
        # print("✅ 最终答案：")
        # print(final_state.values.get("task_result", "无"))

        return log


if __name__ == "__main__":
    from ai_core.until.untils import setup_global_logger
    from ai_core.agent.unit.agent_state import AgentState

    setup_global_logger("../logs/react_agent.log.1")

    config = {"configurable": {"thread_id": "thread_1"}, "recursion_limit": 50}
    with open("../config/onescience_agent.yml", "r") as f:
        agent_config = safe_load(f)
        agent_config["user"] = "abcd"

    # inputs = {"messages": [("user",
    #                        "请帮我生成围绕春这个主题的一段话，总字数不超100字,并写入/work/home/onescience2025/kangdf/agent/molsculptor_agent/agent/demo/test.txt文件中")]}
    inputs = {"messages": [("user", "请使用2个搜索步数帮我设计一个蛋白小分子结构")]}
    react = ReactAgent(agent_config, AgentState)
    react.stream(inputs, config)
