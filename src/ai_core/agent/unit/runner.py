import logging

from langchain_core.runnables import RunnableSequence
from langchain_core.messages import SystemMessage
from langgraph.typing import StateLike
from typing import Dict
from ai_core.llm import ChatModel
from tool_manager import get_tools
from ai_core.rag.rag_server import RagServer


class Runner:
    name: str = "default"
    system_prompt = None
    logger = logging.getLogger(__name__)

    def __init__(self, config: Dict):
        self.user = config.get("user", "default")
        self.tools, self.tool_examples = get_tools(config.get("application", {}))
        self.rag = RagServer()
        chat_model_config = config["chat_model"]
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
