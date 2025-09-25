import logging

from langchain_core.runnables import RunnableSequence
from langchain_core.messages import SystemMessage
from langgraph.typing import StateLike
from typing import Dict, Optional
from agent.llm import ChatModel
from tool_manager import get_tools
from agent.rag.rag_server import RagServer


class Runner:
    name: str = "default"
    system_prompt = None
    logger = logging.getLogger(__name__)

    def __init__(self, agent_config: Dict, rag_config: Optional[Dict] = None):
        self.user = agent_config.get("user", "default")
        self.tools, self.tool_examples = get_tools(agent_config.get("tool_modules", []))
        self.rag = RagServer(rag_config) if rag_config else None
        chat_model_config = agent_config["chat_model"]
        if self.name == "tool" or self.name == "parser":
            self.llm = None
        elif self.name not in chat_model_config:
            self.llm = ChatModel[chat_model_config["default"]["factory_name"]](
                **chat_model_config["default"]["model"], tools=self.tools
            )
        else:
            self.llm = ChatModel[chat_model_config[self.name]["factory_name"]](
                **chat_model_config[self.name]["model"], tools=self.tools
            )

    def node(self, state: StateLike) -> Dict:
        messages = [SystemMessage(self.system_prompt)] + state["messages"]
        response = self.llm.invoke(messages)

        self.logger.info(f"{self.name} response: {response}")
        return {"messages": [response]}
