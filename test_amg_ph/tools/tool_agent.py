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


def create_cancel_leave_request_tool() -> StructuredTool:
    return create_langchain_tool(133)


def create_cancel_medicine_ticket_tool() -> StructuredTool:
    return create_langchain_tool(134)


def create_cancel_pickup_ticket_tool() -> StructuredTool:
    return create_langchain_tool(135)


def create_leave_ticket_tool() -> StructuredTool:
    return create_langchain_tool(136)


def create_medicine_ticket_tool() -> StructuredTool:
    return create_langchain_tool(137)


def create_pickup_ticket_tool() -> StructuredTool:
    return create_langchain_tool(138)


def create_edit_leave_tool() -> StructuredTool:
    return create_langchain_tool(139)


def create_edit_medicine_tool() -> StructuredTool:
    return create_langchain_tool(140)


def create_edit_pickup_tool() -> StructuredTool:
    return create_langchain_tool(141)


def create_get_code_ticket_tool() -> StructuredTool:
    return create_langchain_tool(142)


def create_get_learning_schedule_tool() -> StructuredTool:
    return create_langchain_tool(143)


def create_get_menu_tool() -> StructuredTool:
    return create_langchain_tool(144)


def create_get_student_status_tool() -> StructuredTool:
    return create_langchain_tool(146)


def create_get_ticket_summary_month_tool() -> StructuredTool:
    return create_langchain_tool(147)


def create_get_ticket_summary_quarter_tool() -> StructuredTool:
    return create_langchain_tool(148)


def create_leave_date_tool() -> StructuredTool:
    return create_langchain_tool(149)


def create_medicine_date_tool() -> StructuredTool:
    return create_langchain_tool(150)


def create_pickup_date_tool() -> StructuredTool:
    return create_langchain_tool(151)
