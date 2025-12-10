from typing import TypedDict, Annotated, List
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.graph.message import add_messages


class GenericAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    plan: dict
    pending_tools: List[str]


class SupervisorState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    query: str
    original_query: str
    chat_history: Annotated[list[AnyMessage], add_messages]
    list_tasks: List[dict]
