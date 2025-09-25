from collections.abc import Sequence
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """The state of the agent."""

    # add_messages is a reducer that combines message sequences
    messages: Annotated[Sequence[BaseMessage], add_messages]
