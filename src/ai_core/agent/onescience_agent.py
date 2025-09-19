import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "unit"))

import operator
from yaml import safe_load

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, END

from typing import List, Dict, Annotated, TypedDict
from qa_agent import QaAgent
from react_agent import ReactAgent
from ai_core.agent.unit.agent_state import AgentState
from ai_core.agent.unit.task_class_runner import TaskClasser


class OneScienceAgent:

    def __init__(self, config: Dict):
        memery = InMemorySaver()
        task_class = TaskClasser(config)
        qa_agent = QaAgent(config, AgentState, memery)
        react_agent = ReactAgent(config, AgentState, memery)
        workflow = StateGraph(AgentState)
        workflow.add_node("class", task_class.node)
        workflow.add_node("qa_agent", qa_agent.node)
        workflow.add_node("task_agent", react_agent.node)

        workflow.set_entry_point("class")
        workflow.add_conditional_edges(
            "class", self.class_condition, {"qa": "qa_agent", "task": "task_agent"}
        )
        workflow.add_edge("qa_agent", END)

        self.workflow = workflow.compile(checkpointer=memery)

    @staticmethod
    def class_condition(state):
        return "qa" if state["messages"][-1].content == "simple_qa" else "task"

    def stream(self, inputs, config):
        for s in self.workflow.stream(input=inputs, config=config):
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
    react = OneScienceAgent(agent_config)
    react.stream(inputs, config)
