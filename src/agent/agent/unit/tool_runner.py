import json
from langgraph.typing import StateLike
from langchain_core.messages import ToolMessage
from typing import Dict, Optional
from agent.agent.unit.runner import Runner


class ToolRunner(Runner):
    name = "tool"

    def __init__(self, agent_config: Dict, rag_config: Optional[Dict] = None):
        super().__init__(agent_config, rag_config)
        self.name_to_tools = {tool.name: tool for tool in self.tools}

    def node(self, state: StateLike) -> Dict:
        ai_message = state["messages"][-1]

        outputs = []
        for tool_call in ai_message.tool_calls:
            try:
                func = self.name_to_tools.get(tool_call["name"])
                if func:
                    tool_result = func.invoke(input=tool_call["args"])
                    outputs.append(
                        ToolMessage(
                            content=json.dumps(tool_result, ensure_ascii=False),
                            name=tool_call["name"],
                            tool_call_id=tool_call["id"],
                        )
                    )
            except Exception as e:
                # Handle any errors that occur during tool execution
                outputs.append(
                    ToolMessage(
                        content=json.dumps({"error": str(e)}, ensure_ascii=False),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )
        self.logger.info(f"tool_runner: {outputs}")
        return {"messages": outputs}
