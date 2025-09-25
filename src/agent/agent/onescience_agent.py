#!/user/bin/env
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.join(os.path.dirname(__file__), "unit"))

import uuid
import logging
from agent.untils.until import setup_global_logger
from agent.untils.until import gen_job_log_name
from agent.agent.react_agent import ReactAgent
from agent.agent.unit.agent_state import AgentState
from agent.agent.unit.tool_manager import match_tool_modules


class OnescienceAgent:
    support_llms = ["Qwen3-32B"]

    def __init__(self, job_name: str, llm_server: str, llm: str = "Qwen3-32B"):
        self.job_name = job_name
        if llm not in self.support_llms:
            raise ValueError("Only support Qwen3-32B now")
        self.agent_config = {
            "chat_model": {
                "default": {
                    "factory_name": "LocalQwen3",
                    "model": {
                        "model_name": llm,
                        "model_server": llm_server,
                        "tools": []
                    }
                }
            }
        }
        log_name = gen_job_log_name(job_name, "../logs")
        setup_global_logger(f"../logs/{log_name}")
        self.logger = logging.getLogger(__name__)

    def run(self, input: str):
        user = uuid.uuid4()
        config = {"configurable": {"thread_id": f"thread_{user}"}, "recursion_limit": 50}
        inputs = {"messages": [("user", input)]}
        tool_modules = match_tool_modules(input,
                                          self.agent_config["chat_model"]["default"])
        self.logger.info(f"match tool modules: {tool_modules}")
        if tool_modules and len(tool_modules):
            self.agent_config["tool_modules"] = tool_modules
        react = ReactAgent(agent_config=self.agent_config,
                           rag_config=None,
                           state_schema=AgentState,
                           reflect_times=0)
        return react.stream(inputs, config)

if __name__ == "__main__":
    agent = OnescienceAgent("molsculptor", "http://a02r2n09:8000/v1")
    agent.run(
        "请使用2个搜索步数帮我设计一个蛋白小分子结构，其中各参数初始化的配置文件路径为'../application/molsculptor/config.ini")
