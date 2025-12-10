"""
AgentV7: Hybrid Parallel Tool Calling + Supervisor Task Grouping

Improvements over V5:
1. ✨ Parallel tool calling: Agent can call multiple independent tools at once
2. ✨ Supervisor task grouping: Same agent tasks merged with original query
3. 🚀 Better performance: Reduced LLM calls for independent operations
"""

import json
import requests
from datetime import datetime
from random import random
from utils import euclidean_distance
from typing import TypedDict, Annotated, Literal, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, AnyMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from pydantic import BaseModel, Field

load_dotenv()

# ========================= CONFIG =========================
CITY_IDS = {
    'Hà Nội': 62, 'Hồ Chí Minh': 1, 'Đà Nẵng': 60, 'Hải Phòng': 59,
    'Cần Thơ': 61, 'Đồng Nai': 42, 'Bình Dương': 50, 'Thủ Đức': 1,
}
import math
import json
from typing import List, Dict

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Tính khoảng cách (km) giữa 2 toạ độ bằng công thức Haversine.
    """
    R = 6371.0  # bán kính Trái Đất (km)
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c
# ========================= TOOLS =========================
@tool
def get_near_salon(user_address: str, city: str = "Hà Nội") -> str:
    """Tìm và gợi ý salon 30Shine gần nhất dựa trên địa chỉ và thành phố của khách."""
    addr_lower = user_address.lower()
    if "ba đình" in addr_lower or "ba dinh" in addr_lower:
        user_lat, user_lon = 21.0333, 105.8278
    elif "hoàn kiếm" in addr_lower or "hoan kiem" in addr_lower:
        user_lat, user_lon = 21.0285, 105.8542
    elif "tây hồ" in addr_lower or "tay ho" in addr_lower:
        user_lat, user_lon = 21.0645, 105.8230
    else:
        # Toạ độ trung tâm Hà Nội mặc định (giả)
        user_lat, user_lon = 21.0278, 105.8342

    # === Bước 2: Danh sách salon giả (mock data)
    salons: List[Dict] = [
        {"id": 1, "name": "Salon Hoa Mai", "address": "Số 10, Quận Hoàn Kiếm, Hà Nội", "lat": 21.0290, "lon": 105.8536,
         "rating": 4.6},
        {"id": 2, "name": "Hair Studio 88", "address": "Khu vực Ba Đình, Hà Nội", "lat": 21.0340, "lon": 105.8285,
         "rating": 4.4},
        {"id": 3, "name": "Tóc & Spa Tây Hồ", "address": "Tây Hồ, Hà Nội", "lat": 21.0630, "lon": 105.8210,
         "rating": 4.2},
        {"id": 4, "name": "Salon Minh Châu", "address": "Hai Bà Trưng, Hà Nội", "lat": 21.0125, "lon": 105.8532,
         "rating": 4.1},
        {"id": 5, "name": "Barber Street", "address": "Cầu Giấy, Hà Nội", "lat": 21.0295, "lon": 105.7836,
         "rating": 4.3},
        {"id": 6, "name": "Salon Gội Đầu Thư Giãn", "address": "Đống Đa, Hà Nội", "lat": 21.0115, "lon": 105.8467,
         "rating": 4.0},
    ]

    # === Bước 3: Tính khoảng cách và lọc/sắp xếp
    results = []
    for s in salons:
        dist_km = _haversine(user_lat, user_lon, s["lat"], s["lon"])
        s_copy = s.copy()
        s_copy["distance_km"] = round(dist_km, 3)
        results.append(s_copy)

    # Lọc: chỉ trả salon trong bán kính 8 km (giá trị giả), sắp xếp theo khoảng cách tăng dần
    max_radius_km = 8.0
    nearby = [r for r in results if r["distance_km"] <= max_radius_km]
    nearby.sort(key=lambda x: x["distance_km"])

    # Nếu không tìm thấy salon nào trong radius, trả về top 3 gần nhất (fallback)
    if not nearby:
        results.sort(key=lambda x: x["distance_km"])
        nearby = results[:3]

    # === Bước 4: Đóng gói JSON trả về (chuỗi)
    output = {
        "query_address": user_address,
        "city": city,
        "user_coord": {"lat": user_lat, "lon": user_lon},
        "count": len(nearby),
        "salons": nearby
    }
    return json.dumps(output, ensure_ascii=False, indent=2)

@tool
def check_availability(salon_address: str, date: str, time: str) -> str:
    """Kiểm tra xem khung giờ tại salon có còn trống không."""
    if random() < 0.15:
        return f"Slot {time} ngày {date} tại {salon_address} đã hết ạ. Gần nhất còn: 09:00 và 10:00"
    return f"Slot {time} ngày {date} tại {salon_address} còn trống ạ!"

@tool
def book_appointment(salon_address: str, date: str, time: str, phone: str) -> str:
    """Đặt lịch cắt tóc chính thức cho khách."""
    return f"Đặt lịch thành công cho anh tại {salon_address} – {date} {time} – SĐT {phone}. Em đã gửi tin xác nhận rồi ạ!"

@tool
def cancel_appointment(phone: str) -> str:
    """Hủy lịch hẹn theo số điện thoại."""
    return f"Đã hủy toàn bộ lịch hẹn của số {phone} thành công ạ."

@tool
def list_branches() -> str:
    """Liệt kê tổng quan các chi nhánh 30Shine."""
    return "30Shine hiện có hơn 120 chi nhánh tại Hà Nội, TP.HCM, Đà Nẵng, Hải Phòng, Cần Thơ, Đồng Nai, Bình Dương... Anh muốn em tìm salon ở khu vực nào ạ?"

@tool
def get_info() -> str:
    """Cung cấp thông tin giá cả, dịch vụ, combo của 30Shine."""
    return (
        "Bảng giá tham khảo 30Shine:\n"
        "• Cắt tóc thường: 100k\n"
        "• Cắt + Gội + Massage: 250k\n"
        "• VIP Full service: 450k\n"
        "Anh cần tư vấn thêm gì không ạ?"
    )


# ========================= SCHEMA V7 (NEW) =========================
from langchain_core.utils.function_calling import convert_to_openai_tool

def safe_tool_description(tool_func):
    """Generate safe tool description for prompts"""
    tool_name = getattr(tool_func, 'name', None) or getattr(tool_func, '__name__', str(tool_func))
    try:
        spec = convert_to_openai_tool(tool_func)
        func = spec.get('function', {})
        name = func.get('name', tool_name)
        desc = func.get('description', 'No description')
        params = func.get('parameters', {})
        props = params.get('properties', {})
        if not props:
            param_str = "no params"
        else:
            param_parts = [f"{pname}: {pinfo.get('type', 'any')}" for pname, pinfo in props.items()]
            param_str = ", ".join(param_parts)
        return f"• {name}({param_str}) → {desc}"
    except Exception as e:
        print(f"Tool desc error for {tool_name}: {e}")
        return f"• {tool_name}(params unknown) → Tool for {tool_name}"


class ToolCall(BaseModel):
    """Single tool call specification"""
    tool_name: str = Field(description="Tên tool cần gọi (phải chính xác)")
    tool_args: dict = Field(default_factory=dict, description="Arguments cho tool")


class BookingPlan(BaseModel):
    """
    ✨ V7 NEW: Support parallel tool calling with List[ToolCall]
    """
    thought: str = Field(description="Suy nghĩ chi tiết bằng tiếng Việt về tình huống hiện tại")
    analysis: str = Field(description="Phân tích: đã có gì, còn thiếu gì")
    next_action: Literal["call_tool", "ask_user", "respond"] = Field(
        description="Hành động tiếp theo: call_tool (gọi tool), ask_user (hỏi thêm), respond (trả lời)"
    )
    # ✨ CHANGE: Single → List (support parallel)
    tool_calls: List[ToolCall] = Field(
        default_factory=list,
        description="""
        Danh sách tools cần gọi (có thể 1 hoặc nhiều):
        - [1 tool]: Sequential execution (thường dùng)
        - [2+ tools]: Parallel execution CHỈ KHI chúng KHÔNG phụ thuộc nhau

        Ví dụ parallel hợp lý:
        • "So sánh salon gần tôi với tất cả chi nhánh" → get_near_salon + list_branches
        • "Check lịch ở 2 salon" → 2× check_availability

        Ví dụ PHẢI sequential:
        • "Tìm salon → check lịch → book" (có dependency)
        """
    )
    response: str = Field(default="", description="Câu trả lời cho user (nếu next_action=ask_user hoặc respond)")


class InfoPlan(BaseModel):
    """✨ V7 NEW: Info plan with parallel support"""
    thought: str = Field(description="Suy nghĩ về câu hỏi của khách")
    next_action: Literal["call_tool", "respond"] = Field(description="Hành động tiếp theo")
    tool_calls: List[ToolCall] = Field(default_factory=list, description="Tools cần gọi (có thể nhiều)")
    response: str = Field(default="", description="Câu trả lời nếu next_action=respond")


# ========================= PARALLEL TOOL EXECUTOR =========================
def _execute_single_tool(tool_call: dict, tool_map: dict) -> str:
    """
    Helper: Execute 1 tool và return formatted result
    """
    tool_name = tool_call.get("tool_name", "")
    tool_args = tool_call.get("tool_args") or {}

    if tool_name not in tool_map:
        return f"❌ [{tool_name}] Tool không tồn tại. Available: {list(tool_map.keys())}"

    if not tool_args and tool_name not in ["get_info", "list_branches"]:
        return f"⚠️ [{tool_name}] Thiếu arguments"

    try:
        tool = tool_map[tool_name]
        result = tool.invoke(tool_args)
        print(f"[TOOL_EXECUTOR] ✅ {tool_name}: Success")
        return f"✅ [{tool_name}]:\n{result}"
    except Exception as e:
        error_msg = f"❌ [{tool_name}] Error: {str(e)}"
        print(f"[TOOL_EXECUTOR] {error_msg}")
        return error_msg


def parallel_tool_executor(tool_calls: List[dict], tool_map: dict) -> str:
    """
    ✨ V7 NEW: Execute multiple tools in parallel using ThreadPoolExecutor

    Args:
        tool_calls: List of {tool_name, tool_args}
        tool_map: Dict mapping tool name to tool function

    Returns:
        Combined result string
    """
    if not tool_calls:
        return "[No tools to execute]"

    num_tools = len(tool_calls)
    print(f"\n[TOOL_EXECUTOR] 🔧 Executing {num_tools} tool(s)")

    # ============ SINGLE TOOL (Fast path) ============
    if num_tools == 1:
        return _execute_single_tool(tool_calls[0], tool_map)

    # ============ MULTIPLE TOOLS (Parallel) ============
    print(f"[TOOL_EXECUTOR] ⚡ Parallel execution mode")

    results = []
    with ThreadPoolExecutor(max_workers=min(num_tools, 5)) as executor:
        # Submit all tasks
        future_to_tool = {
            executor.submit(_execute_single_tool, tc, tool_map): tc
            for tc in tool_calls
        }

        # Collect results as they complete
        for future in as_completed(future_to_tool):
            tool_call = future_to_tool[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                tool_name = tool_call.get("tool_name", "unknown")
                results.append(f"❌ [{tool_name}] Exception: {e}")

    # Combine all results
    combined = "\n\n".join(results)
    return f"[Tool Results - Parallel Execution]:\n{combined}"


# ========================= BOOKING AGENT V7 =========================
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

booking_tools = [get_near_salon, check_availability, book_appointment, cancel_appointment, list_branches]
TOOL_DESCRIPTIONS = "\n".join([safe_tool_description(t) for t in booking_tools])
print("TOOL_DESCRIPTIONS generated:", TOOL_DESCRIPTIONS)


class BookingState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    plan: dict


PARALLEL_GUIDELINES = """
🔀 PARALLEL TOOL CALLING (V7 NEW):
Bạn có thể gọi NHIỀU tools cùng lúc trong tool_calls CHỈ KHI:
✅ Chúng KHÔNG phụ thuộc nhau (independent)
✅ Chúng CẦN THIẾT cùng lúc để trả lời query

📌 Ví dụ NÊN dùng parallel:
• Query: "So sánh salon gần tôi với tất cả chi nhánh"
  → tool_calls: [
      {{"tool_name": "get_near_salon", "tool_args": {{"user_address": "...", "city": "Hà Nội"}}}},
      {{"tool_name": "list_branches", "tool_args": {{}}}}
    ]

• Query: "Kiểm tra lịch trống ở salon A và salon B"
  → tool_calls: [
      {{"tool_name": "check_availability", "tool_args": {{"salon_address": "A", ...}}}},
      {{"tool_name": "check_availability", "tool_args": {{"salon_address": "B", ...}}}}
    ]

❌ Ví dụ KHÔNG nên parallel (phải sequential):
• Query: "Tìm salon gần tôi rồi book lịch"
  → Lần 1: tool_calls: [{{"tool_name": "get_near_salon", ...}}]
  → Đợi kết quả
  → Lần 2: tool_calls: [{{"tool_name": "check_availability", ...}}]
  → Lần 3: tool_calls: [{{"tool_name": "book_appointment", ...}}]

🎯 Nguyên tắc: Khi nghi ngờ → Gọi tuần tự (an toàn hơn)
"""


def booking_planner(state: BookingState):
    """
    ✨ V7 Planner with parallel tool calling support
    """
    messages = state["messages"]

    system_prompt = f"""
Today is: {datetime.now().strftime("%d/%m/%Y %H:%M")}

Bạn là Janie – trợ lý đặt lịch 30Shine thông minh. Nhiệm vụ: phân tích và lên kế hoạch.

=== TOOLS CÓ SẴN ===
{TOOL_DESCRIPTIONS}


=== QUY TRÌNH SUY NGHĨ ===
1. Đọc kỹ tin nhắn của khách
2. Phân tích: đã có thông tin gì? còn thiếu gì?
3. Quyết định hành động tiếp theo:
   - "call_tool": Nếu đủ thông tin để gọi tool → điền tool_calls (có thể 1 hoặc nhiều)
   - "ask_user": Nếu thiếu thông tin quan trọng → điền response (câu hỏi)
   - "respond": Nếu đã hoàn thành hoặc chỉ cần trả lời → điền response

=== QUY TẮC QUAN TRỌNG ===
- Nếu có địa chỉ (tên đường + thành phố) → GỌI get_near_salon ngay, KHÔNG hỏi thêm
- Nếu khách muốn đặt lịch nhưng chưa có salon → tìm salon trước
- Nếu có salon + ngày giờ → check_availability
- Nếu slot OK + có SĐT → book_appointment
- Phong cách response: xưng em, gọi anh, kết thúc bằng "ạ"
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{messages}")
    ])

    chain = prompt | llm.with_structured_output(BookingPlan, method="function_calling")

    try:
        plan: BookingPlan = chain.invoke({"messages": messages})
        print(f"\n[PLANNER] 💭 Thought: {plan.thought}")
        print(f"[PLANNER] 📊 Analysis: {plan.analysis}")
        print(f"[PLANNER] ➡️ Next Action: {plan.next_action}")
        if plan.tool_calls:
            print(f"[PLANNER] 🔧 Tools ({len(plan.tool_calls)}): {[tc.tool_name for tc in plan.tool_calls]}")
        if plan.response:
            print(f"[PLANNER] 💬 Response: {plan.response[:100]}...")
    except Exception as e:
        print(f"[PLANNER] ❌ Error: {e}")
        plan = BookingPlan(
            thought="Lỗi parse, cần hỏi lại",
            analysis="Không parse được",
            next_action="ask_user",
            response="Dạ anh ơi, em chưa hiểu lắm, anh nói rõ hơn được không ạ?"
        )

    return {
        "messages": [AIMessage(content=f"[Plan] {plan.thought}", name="planner")],
        "plan": plan.model_dump()
    }


def tool_executor(state: BookingState):
    """
    ✨ V7 Tool executor with parallel support
    """
    plan = state.get("plan", {})
    tool_calls = plan.get("tool_calls") or []

    tool_map = {t.name: t for t in booking_tools}
    result = parallel_tool_executor(tool_calls, tool_map)

    return {
        "messages": [HumanMessage(content=result)]
    }


def booking_responder(state: BookingState):
    """Tạo response cuối cùng từ plan.response"""
    print("\n" + "="*50)
    print("[BOOKING_RESPONDER] ▶ Tạo response")

    plan = state.get("plan", {})
    response = plan.get("response", "")

    if response:
        text = response
        print(f"[BOOKING_RESPONDER] ✅ Dùng response từ plan: {text[:100]}...")
    else:
        print("[BOOKING_RESPONDER] 📝 Tổng hợp từ conversation...")
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Tổng hợp lại cuộc trò chuyện thành câu trả lời tự nhiên. Xưng em, gọi anh, kết thúc bằng ạ."),
            ("placeholder", "{messages}")
        ])
        text = (prompt | llm).invoke({"messages": state["messages"]}).content

    print(f"[BOOKING_RESPONDER] 📝 Final: {text}")
    print("="*50)
    return {"messages": [AIMessage(content=text, name="booking_node")]}


def booking_router(state: BookingState):
    """Router dựa trên plan.next_action"""
    print("\n" + "-"*30)
    print("[BOOKING_ROUTER] 🔀 Routing...")

    plan = state.get("plan", {})
    next_action = plan.get("next_action", "respond")

    print(f"[BOOKING_ROUTER] next_action = {next_action}")

    if next_action == "call_tool":
        print("[BOOKING_ROUTER] → tool_executor")
        return "tool_executor"
    else:
        print("[BOOKING_ROUTER] → respond")
        return "respond"


def create_booking_agent():
    """Build booking agent graph with V7 parallel support"""
    g = StateGraph(BookingState)

    g.add_node("planner", booking_planner)
    g.add_node("tool_executor", tool_executor)
    g.add_node("respond", booking_responder)

    g.add_edge(START, "planner")
    g.add_conditional_edges(
        "planner",
        booking_router,
        {"tool_executor": "tool_executor", "respond": "respond"}
    )
    g.add_edge("tool_executor", "planner")
    g.add_edge("respond", END)

    return g.compile(checkpointer=MemorySaver())


# ========================= INFORMATION AGENT V7 =========================
info_tools = [get_info, list_branches, get_near_salon]


def info_planner(state):
    print("\n" + "="*50)
    print("[INFO_PLANNER] ▶ Planning")

    info_tool_desc = "\n".join([safe_tool_description(t) for t in info_tools])

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""Bạn là Janie chuyên tư vấn dịch vụ 30Shine.

=== TOOLS ===
{info_tool_desc}

{PARALLEL_GUIDELINES}

Nếu biết câu trả lời → next_action="respond", điền response
Nếu cần tra cứu → next_action="call_tool", điền tool_calls (có thể nhiều nếu independent)

Phong cách: xưng em, gọi anh, kết thúc bằng ạ."""),
        ("placeholder", "{messages}")
    ])

    try:
        plan = (prompt | llm.with_structured_output(InfoPlan, method="function_calling")).invoke({"messages": state["messages"]})
        print(f"[INFO_PLANNER] 💭 {plan.thought}")
        print(f"[INFO_PLANNER] ➡️ {plan.next_action}")
        if plan.tool_calls:
            print(f"[INFO_PLANNER] 🔧 Tools: {[tc.tool_name for tc in plan.tool_calls]}")
    except Exception as e:
        print(f"[INFO_PLANNER] ❌ Error: {e}")
        plan = InfoPlan(thought="Lỗi", next_action="respond", response="Dạ anh ơi, em chưa hiểu, anh hỏi lại nhé ạ!")

    return {
        "messages": [AIMessage(content=f"[Info Plan] {plan.thought}", name="info_planner")],
        "plan": plan.model_dump()
    }


def info_tool_executor(state):
    plan = state.get("plan", {})
    tool_calls = plan.get("tool_calls") or []

    tool_map = {t.name: t for t in info_tools}
    result = parallel_tool_executor(tool_calls, tool_map)

    return {"messages": [HumanMessage(content=result)]}


def info_responder(state):
    plan = state.get("plan", {})
    response = plan.get("response", "")

    if not response:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Tổng hợp câu trả lời tự nhiên. Xưng em, gọi anh, kết thúc bằng ạ."),
            ("placeholder", "{messages}")
        ])
        response = (prompt | llm).invoke({"messages": state["messages"]}).content

    print(f"[INFO_RESPONDER] 📝 {response}")
    return {"messages": [AIMessage(content=response, name="information_node")]}


def info_router(state):
    plan = state.get("plan", {})
    if plan.get("next_action") == "call_tool":
        return "tool_executor"
    return "respond"


def create_information_agent():
    g = StateGraph(BookingState)
    g.add_node("planner", info_planner)
    g.add_node("tool_executor", info_tool_executor)
    g.add_node("respond", info_responder)

    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", info_router, {"tool_executor": "tool_executor", "respond": "respond"})
    g.add_edge("tool_executor", "planner")
    g.add_edge("respond", END)

    return g.compile(checkpointer=MemorySaver())


# ========================= SUPERVISOR V7 (WITH TASK GROUPING) =========================
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    query: str
    original_query: str  # ✨ NEW: Keep original query for task grouping
    chat_history: Annotated[list[AnyMessage], add_messages]
    list_tasks: List[dict]


class Action(BaseModel):
    name: Literal["booking_node", "information_node"]
    query: str = Field(..., description="Query gửi cho agent này")


class SupervisorPlan(BaseModel):
    actions: List[Action]


SUPERVISOR_SYSTEM_PROMPT = """
Bạn là Supervisor điều phối 2 agent chuyên biệt:

booking_node → Đặt lịch, hủy lịch, check slot, tìm salon gần nhất
information_node → Tư vấn giá, dịch vụ, combo, tiện ích, chỗ để xe, v.v.

Phân tích query và trả về đúng JSON theo schema.
"""

booking_agent = create_booking_agent()
information_agent = create_information_agent()


def group_tasks_by_agent(actions: List[Action], original_query: str) -> List[dict]:
    """
    ✨ V7 NEW: Group multiple tasks for same agent into 1 task with original query

    Args:
        actions: List of actions from supervisor plan
        original_query: Original user query

    Returns:
        List of deduplicated tasks [{"name": agent_name, "query": original_query, "status": "pending"}]

    Example:
        Input: [
            {name: "booking_node", query: "tìm salon"},
            {name: "booking_node", query: "check lịch"},
            {name: "information_node", query: "giá"}
        ]

        Output: [
            {name: "booking_node", query: "<original_query>", status: "pending"},
            {name: "information_node", query: "<original_query>", status: "pending"}
        ]
    """
    # Deduplicate by agent name
    seen_agents = set()
    grouped_tasks = []

    for action in actions:
        if action.name not in seen_agents:
            seen_agents.add(action.name)
            grouped_tasks.append({
                "name": action.name,
                "query": original_query,  # ✨ Use original query instead of split queries
                "status": "pending"
            })

    return grouped_tasks


def supervisor_node(state: AgentState):
    print("\n" + "#"*60)
    print("[SUPERVISOR] 🎯 Supervisor đang xử lý...")
    print(f"[SUPERVISOR] Current tasks: {state.get('list_tasks', [])}")

    # Nếu đã có task đang chạy → tiếp tục
    for task in state.get("list_tasks", []):
        if task.get("status") != "done":
            print(f"[SUPERVISOR] → Tiếp tục task chưa hoàn thành: {task['name']}")
            print("#"*60)
            return Command(goto=task["name"], update={"query": task["query"]})

    # Nếu tất cả done → kết thúc
    if state.get("list_tasks") and all(t.get("status") == "done" for t in state["list_tasks"]):
        print("[SUPERVISOR] ✅ Tất cả tasks hoàn thành → END")
        print("#"*60)
        return Command(goto=END)

    # Phân tích query mới
    print("[SUPERVISOR] 🔍 Phân tích query mới...")
    messages = [{"role": "system", "content": SUPERVISOR_SYSTEM_PROMPT}] + state["messages"]
    response = llm.with_structured_output(SupervisorPlan).invoke(messages)

    # ✨ V7 NEW: Group tasks by agent + use original query
    original_query = state.get("original_query") or state.get("query", "")
    tasks = group_tasks_by_agent(response.actions, original_query)

    print(f"[SUPERVISOR] 📋 Raw actions: {[a.name for a in response.actions]}")
    print(f"[SUPERVISOR] ✨ Grouped tasks: {[t['name'] for t in tasks]}")
    for i, t in enumerate(tasks):
        print(f"[SUPERVISOR]   {i+1}. {t['name']}: {t['query'][:50]}...")

    # Chọn task đầu tiên
    first_task = tasks[0]
    print(f"[SUPERVISOR] → Chuyển đến: {first_task['name']}")
    print("#"*60)
    return Command(
        goto=first_task["name"],
        update={"list_tasks": tasks, "query": first_task["query"]}
    )


def booking_node(state: AgentState):
    print("\n" + "="*60)
    print("[BOOKING_NODE] 📅 Bắt đầu xử lý booking...")
    print(f"[BOOKING_NODE] Query: {state['query']}")

    result = booking_agent.invoke({"messages": state["chat_history"] + [HumanMessage(content=state["query"])]})
    final_msg = result["messages"][-1].content

    print(f"[BOOKING_NODE] 📝 Kết quả: {final_msg[:100]}...")

    # Đánh dấu task done
    for t in state["list_tasks"]:
        if t["name"] == "booking_node":
            t["status"] = "done"
            print("[BOOKING_NODE] ✅ Đánh dấu task done")

    print("[BOOKING_NODE] → Quay về supervisor")
    print("="*60)
    return Command(update={"messages": [AIMessage(content=final_msg, name="booking_node")]}, goto="supervisor")


def information_node(state: AgentState):
    print("\n" + "="*60)
    print("[INFO_NODE] ℹ️ Bắt đầu xử lý thông tin...")
    print(f"[INFO_NODE] Query: {state['query']}")

    result = information_agent.invoke({"messages": state["chat_history"] + [HumanMessage(content=state["query"])]})
    final_msg = result["messages"][-1].content

    print(f"[INFO_NODE] 📝 Kết quả: {final_msg[:100]}...")

    for t in state["list_tasks"]:
        if t["name"] == "information_node":
            t["status"] = "done"
            print("[INFO_NODE] ✅ Đánh dấu task done")

    print("[INFO_NODE] → Quay về supervisor")
    print("="*60)
    return Command(update={"messages": [AIMessage(content=final_msg, name="information_node")]}, goto="supervisor")


# Build final graph
workflow = StateGraph(AgentState)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("booking_node", booking_node)
workflow.add_node("information_node", information_node)
workflow.add_edge(START, "supervisor")
app = workflow.compile(checkpointer=MemorySaver())


# ========================= STREAMLIT UI =========================
st.set_page_config(page_title="30Shine AgentV7", layout="wide")
st.title("🚀 30Shine AgentV7: Parallel Tools + Task Grouping")
st.caption("✨ New: Parallel tool calling | Smart supervisor task grouping")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "30shine_v7_001"

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

# User input
if prompt := st.chat_input("Anh ơi, em nghe nè ạ..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.spinner("Janie đang suy nghĩ..."):
        config = {"configurable": {"thread_id": st.session_state.thread_id}}

        inputs = {
            "messages": [HumanMessage(content=prompt)],
            "query": prompt,
            "original_query": prompt,  # ✨ NEW: Keep original for task grouping
            "chat_history": st.session_state.chat_history,
            "list_tasks": []
        }

        final_answer = None
        try:
            for chunk in app.stream(inputs, config=config, stream_mode="values"):
                msgs = chunk.get("messages", [])
                if msgs and msgs[-1].content:
                    last_msg = msgs[-1]
                    if hasattr(last_msg, "name") and last_msg.name in ["booking_node", "information_node"]:
                        final_answer = last_msg.content
                    elif len(msgs) == 1 and not hasattr(last_msg, "name"):
                        final_answer = last_msg.content

            if final_answer:
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
                st.session_state.chat_history.append(HumanMessage(content=prompt))
                st.session_state.chat_history.append(AIMessage(content=final_answer))
                st.chat_message("assistant").write(final_answer)
            else:
                fallback = "Dạ anh ơi, em chưa hiểu lắm, anh nói lại giúp em được không ạ?"
                st.session_state.messages.append({"role": "assistant", "content": fallback})
                st.chat_message("assistant").write(fallback)

        except Exception as e:
            st.error(f"Lỗi hệ thống: {e}")
            st.write("Chi tiết lỗi:", e)
