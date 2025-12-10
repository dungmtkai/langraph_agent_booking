from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class ToolSetting:
    id: int
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    flow_data: Dict[str, Any] = field(default_factory=dict)


CANCEL_LEAVE_REQUEST = ToolSetting(
    id=133,
    name="cancel_leave_request",
    description="Hủy thông tin nghỉ phép",
    parameters={
        "type": "object",
        "properties": {
            "old_dates": {
                "type": "array",
                "items": {"type": "string", "pattern": r"^\d{2}-\d{2}-\d{4}$"},
                "description": "Danh sách các ngày nghỉ muốn hủy. Định dạng dd-MM-yyyy"
            }
        },
        "required": ["old_dates"]
    }
)

CANCEL_MEDICINE_TICKET = ToolSetting(
    id=134,
    name="cancel_medicine_ticket",
    description="Hủy đơn (ticket) đã tạo",
    parameters={
        "type": "object",
        "properties": {
            "old_dates": {
                "type": "array",
                "items": {"type": "string", "pattern": r"^\d{2}-\d{2}-\d{4}$"},
                "description": "Danh sách các ngày dặn thuốc muốn hủy. Định dạng dd-MM-yyyy"
            }
        },
        "required": ["old_dates"]
    }
)

CANCEL_PICKUP_TICKET = ToolSetting(
    id=135,
    name="cancel_pickup_ticket",
    description="Hủy đơn (ticket) dặn đón",
    parameters={
        "type": "object",
        "properties": {
            "old_dates": {
                "type": "array",
                "items": {"type": "string", "pattern": r"^\d{2}-\d{2}-\d{4}$"},
                "description": "Danh sách các ngày dặn đón muốn hủy. Định dạng dd-MM-yyyy"
            }
        },
        "required": ["old_dates"]
    }
)

CREATE_LEAVE_TICKET = ToolSetting(
    id=136,
    name="create_leave_ticket",
    description="Tạo yêu cầu nghỉ phép cho bé",
    parameters={
        "type": "object",
        "properties": {
            "reason": {"type": "string", "description": "Lý do xin nghỉ"},
            "start_leave_datetime": {"type": "string", "description": "Ngày bắt đầu nghỉ"},
            "number_of_days": {"type": "integer", "description": "Số ngày nghỉ"},
            "dates": {"type": "array", "items": {"type": "string"}, "description": "Danh sách ngày nghỉ"}
        },
        "required": ["reason"]
    }
)

CREATE_MEDICINE_TICKET = ToolSetting(
    id=137,
    name="create_medicine_ticket",
    description="Dặn dò giáo viên, đăng kí uống thuốc cho con",
    parameters={
        "type": "object",
        "properties": {
            "dates": {"type": "array", "items": {"type": "string"}, "description": "Danh sách ngày dặn thuốc"},
            "parent_note": {"type": "string", "description": "Ghi chú của phụ huynh"},
            "medicine_images": {"type": "array", "items": {"type": "string"}, "description": "Ảnh thuốc"}
        },
        "required": ["dates"]
    }
)

CREATE_PICKUP_TICKET = ToolSetting(
    id=138,
    name="create_pickup_ticket",
    description="Hỗ trợ phụ huynh tạo ticket dặn dò người đón bé",
    parameters={
        "type": "object",
        "properties": {
            "dates": {"type": "array", "items": {"type": "string"}, "description": "Danh sách ngày dặn đón"},
            "pickup_instructions": {"type": "string", "description": "Hướng dẫn đón"},
            "notes": {"type": "string", "description": "Ghi chú"}
        },
        "required": []
    }
)

EDIT_SUBMITTED_TICKET_LEAVE = ToolSetting(
    id=139,
    name="edit_submitted_ticket_leave",
    description="Sửa thông tin ticket đã tạo: ticket nghỉ học",
    parameters={
        "type": "object",
        "properties": {
            "old_dates": {"type": "array", "items": {"type": "string"}, "description": "Ngày cũ cần sửa"},
            "dates": {"type": "array", "items": {"type": "string"}, "description": "Ngày mới"},
            "reason": {"type": "string", "description": "Lý do mới"}
        },
        "required": ["old_dates"]
    }
)

EDIT_SUBMITTED_TICKET_MEDICINE = ToolSetting(
    id=140,
    name="edit_submitted_ticket_medicine",
    description="Sửa thông tin ticket đã tạo: ticket dặn thuốc",
    parameters={
        "type": "object",
        "properties": {
            "old_dates": {"type": "array", "items": {"type": "string"}, "description": "Ngày cũ cần sửa"},
            "dates": {"type": "array", "items": {"type": "string"}, "description": "Ngày mới"},
            "parent_note": {"type": "string", "description": "Ghi chú mới"},
            "medicine_images": {"type": "array", "items": {"type": "string"}, "description": "Ảnh thuốc mới"}
        },
        "required": ["old_dates"]
    }
)

EDIT_SUBMITTED_TICKET_PICKUP = ToolSetting(
    id=141,
    name="edit_submitted_ticket_pickup",
    description="Sửa thông tin ticket đã tạo: ticket dặn đón",
    parameters={
        "type": "object",
        "properties": {
            "old_dates": {"type": "array", "items": {"type": "string"}, "description": "Ngày cũ cần sửa"},
            "dates": {"type": "array", "items": {"type": "string"}, "description": "Ngày mới"},
            "pickup_instructions": {"type": "string", "description": "Hướng dẫn đón mới"}
        },
        "required": ["old_dates"]
    }
)

GET_CODE_TICKET = ToolSetting(
    id=142,
    name="get_code_ticket",
    description="Lấy thông tin các đơn đã tạo, lấy mã đơn (khi chỉnh sửa)",
    parameters={}
)

GET_LEARNING_SCHEDULE = ToolSetting(
    id=143,
    name="get_learning_schedule",
    description="Lấy thông tin thời khóa biểu học tập của học sinh",
    parameters={}
)

GET_MENU = ToolSetting(
    id=144,
    name="get_menu",
    description="Cung cấp thực đơn ăn uống của bé theo tuần, ngày",
    parameters={}
)

GET_STUDENT_STATUS = ToolSetting(
    id=146,
    name="get_student_status",
    description="Hỗ trợ lấy thông tin về tình trạng của học sinh (hoạt động, ăn uống, ngủ, vệ sinh)",
    parameters={}
)

GET_TICKET_SUMMARY_BY_MONTH = ToolSetting(
    id=147,
    name="get_ticket_summary_by_month",
    description="Lấy thông tin tổng hợp số lượng ticket theo tháng",
    parameters={}
)

GET_TICKET_SUMMARY_BY_QUARTER = ToolSetting(
    id=148,
    name="get_ticket_summary_by_quarter",
    description="Lấy thông tin tổng hợp số lượng ticket theo quý",
    parameters={}
)

LEAVE_DATE = ToolSetting(
    id=149,
    name="leave_date",
    description="Kiểm tra thời gian xin nghỉ",
    parameters={
        "type": "object",
        "properties": {
            "dates": {"type": "array", "items": {"type": "string"}, "description": "Mảng ngày cần kiểm tra"}
        },
        "required": []
    }
)

MEDICINE_DATE = ToolSetting(
    id=150,
    name="medicine_date",
    description="Kiểm tra thời gian dặn thuốc",
    parameters={
        "type": "object",
        "properties": {
            "dates": {"type": "array", "items": {"type": "string"}, "description": "Mảng ngày cần kiểm tra"}
        },
        "required": []
    }
)

PICKUP_DATE = ToolSetting(
    id=151,
    name="pickup_date",
    description="Kiểm tra thời gian dặn đón",
    parameters={
        "type": "object",
        "properties": {
            "dates": {"type": "array", "items": {"type": "string"}, "description": "Mảng ngày cần kiểm tra"}
        },
        "required": []
    }
)


TOOL_SETTINGS_REGISTRY = {
    133: CANCEL_LEAVE_REQUEST,
    134: CANCEL_MEDICINE_TICKET,
    135: CANCEL_PICKUP_TICKET,
    136: CREATE_LEAVE_TICKET,
    137: CREATE_MEDICINE_TICKET,
    138: CREATE_PICKUP_TICKET,
    139: EDIT_SUBMITTED_TICKET_LEAVE,
    140: EDIT_SUBMITTED_TICKET_MEDICINE,
    141: EDIT_SUBMITTED_TICKET_PICKUP,
    142: GET_CODE_TICKET,
    143: GET_LEARNING_SCHEDULE,
    144: GET_MENU,
    146: GET_STUDENT_STATUS,
    147: GET_TICKET_SUMMARY_BY_MONTH,
    148: GET_TICKET_SUMMARY_BY_QUARTER,
    149: LEAVE_DATE,
    150: MEDICINE_DATE,
    151: PICKUP_DATE,
}

TOOL_SETTINGS_BY_NAME = {
    setting.name: setting for setting in TOOL_SETTINGS_REGISTRY.values()
}


def get_tool_setting(tool_id: int) -> ToolSetting:
    return TOOL_SETTINGS_REGISTRY.get(tool_id)
