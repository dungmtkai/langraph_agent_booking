from langchain_core.tools import StructuredTool
from .settings import ToolSetting, get_tool_setting
from .tool_flow import TOOL_FUNCTIONS


def create_langchain_tool(tool_id: int) -> StructuredTool:
    setting = get_tool_setting(tool_id)
    if not setting:
        raise ValueError(f"Tool with id {tool_id} not found")

    func = TOOL_FUNCTIONS.get(tool_id)
    if not func:
        raise ValueError(f"Function for tool {tool_id} not found")

    return StructuredTool.from_function(
        func=func,
        name=setting.name,
        description=setting.description,
    )
