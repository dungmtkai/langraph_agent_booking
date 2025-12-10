from pydantic import BaseModel, Field
from typing import Literal, List


class ToolSelection(BaseModel):
    tool_name: str = Field(description="Tên tool cần gọi (phải chính xác)")


class GenericAgentPlan(BaseModel):
    thought: str = Field(description="Phân tích đã có gì thiếu gì và cần làm gì tiếp theo (tối đa 2 câu)")
    next_action: Literal["call_tool", "ask_user", "respond"] = Field(
        description="Hành động tiếp theo: call_tool (gọi tool), ask_user (hỏi thêm), respond (trả lời)"
    )
    selected_tools: List[ToolSelection] = Field(
        default=[],
        description="Danh sách tools cần gọi (có thể chọn nhiều tools để gọi song song). Chỉ điền khi next_action=call_tool"
    )
    response: str = Field(
        default="",
        description="Câu trả lời cho user (nếu next_action=ask_user hoặc respond)"
    )


AGENT_NODE_NAMES = Literal[
    "absence_request_node",
    "daily_report_node",
    "feedback_node",
    "get_submitted_ticket_node",
    "learning_schedule_node",
    "meal_info_node",
    "medication_instruction_node",
    "pickup_authorization_node"
]


class Action(BaseModel):
    name: AGENT_NODE_NAMES = Field(description="Tên agent node cần gọi")
    query: str = Field(..., description="Query gửi cho agent này")


class SupervisorPlan(BaseModel):
    actions: List[Action] = Field(
        default=[],
        description="Danh sách actions cần thực hiện. Có thể chọn nhiều agents nếu query có nhiều yêu cầu"
    )
