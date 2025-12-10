from datetime import datetime
from typing import List, Dict, Any, Optional

from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from jinja2 import Template


def safe_tool_description(tool: StructuredTool) -> str:
    tool_name = tool.name
    try:
        spec = convert_to_openai_tool(tool)
        func = spec.get('function', {})
        name = func.get('name', tool_name)
        desc = func.get('description', 'No description')
        params = func.get('parameters', {})
        props = params.get('properties', {})
        if not props:
            param_str = "no params"
        else:
            param_parts = []
            for pname, pinfo in props.items():
                ptype = pinfo.get('type', 'any')
                param_parts.append(f"{pname}: {ptype}")
            param_str = ", ".join(param_parts)
        return f"- {name}({param_str}) -> {desc}"
    except Exception:
        return f"- {tool_name}(params unknown) -> Tool for {tool_name}"


def get_tool_descriptions(tools: List[StructuredTool]) -> str:
    return "\n".join([safe_tool_description(t) for t in tools])


def render_description(description_template: str, user_info: Optional[Dict[str, Any]] = None) -> str:
    template = Template(description_template)
    return template.render(user_info=user_info or {})


PLANNER_SYSTEM_PROMPT_TEMPLATE = """Today is: {current_time}
{description}

=== EXECUTION RULES ===
{rule}

=== INSTRUCTIONS ===
{instruction}

=== TOOLS CÓ SẴN ===
{tool_descriptions}

=== QUY TRÌNH SUY NGHĨ ===
1. Đọc kĩ tin nhắn và lịch sử chat của khách hàng
2. Phân tích: đã có thông tin gì? còn thiếu gì và nên làm gì tiếp theo
3. Quyết định hành động tiếp theo:
   - "call_tool": Nếu cần gọi tool → điền selected_tools (CHỈ CẦN tool_name)
   - "ask_user": Nếu thiếu thông tin quan trọng → điền response (câu hỏi)
   - "respond": Nếu đã hoàn thành hoặc chỉ cần trả lời → điền response

=== QUY TẮC QUAN TRỌNG ===
- selected_tools có thể là danh sách các tool nếu có thể thực hiện song song
- Phong cách response: xưng em, gọi anh, kết thúc bằng "ạ"
"""


def create_planner_system_prompt(
    description: str,
    tool_descriptions: str,
    instruction: str,
    rule: str
) -> str:
    return PLANNER_SYSTEM_PROMPT_TEMPLATE.format(
        current_time=datetime.now().strftime("%d/%m/%Y %H:%M"),
        description=description,
        tool_descriptions=tool_descriptions,
        instruction=instruction,
        rule=rule
    )


TOOL_EXECUTOR_SYSTEM_PROMPT_TEMPLATE = """Bạn là assistant trích xuất thông tin từ cuộc hội thoại để gọi tools.
Hãy phân tích cuộc hội thoại và gọi các tools phù hợp với thông tin có sẵn.

Today is: {current_time}

QUAN TRỌNG:
- Bạn PHẢI gọi các tools sau: {pending_tools}
- Nếu thiếu thông tin bắt buộc, hãy dùng giá trị mặc định hợp lý hoặc suy luận từ context
- Với ngày/giờ: nếu user nói "mai", "chiều nay", etc. → convert sang format cụ thể
- Với địa chỉ: lấy từ context hoặc dùng thông tin user đã cung cấp
"""


def create_tool_executor_system_prompt(pending_tools: List[str]) -> str:
    return TOOL_EXECUTOR_SYSTEM_PROMPT_TEMPLATE.format(
        current_time=datetime.now().strftime("%d/%m/%Y %H:%M"),
        pending_tools=pending_tools
    )


RESPONDER_SYSTEM_PROMPT = "Tổng hợp lại cuộc trò chuyện thành câu trả lời tự nhiên. Xưng em, gọi anh, kết thúc bằng ạ."


AGENT_NODE_TO_ID = {
    "absence_request_node": 159,
    "daily_report_node": 160,
    "feedback_node": 161,
    "get_submitted_ticket_node": 162,
    "learning_schedule_node": 163,
    "meal_info_node": 164,
    "medication_instruction_node": 165,
    "pickup_authorization_node": 166,
}


SUPERVISOR_SYSTEM_PROMPT_TEMPLATE = """Bạn là Supervisor phân tích yêu cầu của phụ huynh và chọn agent phù hợp để xử lý.

## Agents có sẵn:
{agent_descriptions}

## Quy tắc:
- Phân tích ý định chính của user
- Chọn agent phù hợp nhất với yêu cầu
- Có thể chọn nhiều agents nếu query có nhiều yêu cầu riêng biệt
- Mỗi action phải có name (tên agent node) và query (nội dung gửi cho agent)
"""


def build_supervisor_system_prompt() -> str:
    """Build supervisor system prompt dynamically from agent settings.role"""
    from .generic_agent_settings import get_agent_setting

    agent_descriptions = []

    for node_name, agent_id in AGENT_NODE_TO_ID.items():
        setting = get_agent_setting(agent_id)
        if setting:
            agent_descriptions.append(f"{node_name} - {setting.role}")

    return SUPERVISOR_SYSTEM_PROMPT_TEMPLATE.format(
        agent_descriptions="\n".join(agent_descriptions)
    )
