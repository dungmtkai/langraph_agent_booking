"""
30Shine Agent V6 - Architecture Update
=====================================
Key Changes from V5:
1. Plan node: Chỉ lên kế hoạch và chọn tool(s), KHÔNG extract arguments
2. Hỗ trợ chọn NHIỀU tools để thực hiện song song
3. Execute node: Gọi ChatGPT với function calling để extract args và execute

Flow: START → planner → [router] → tool_executor (function calling) → planner (loop)
                           ↓
                        respond → END
"""

import json
import requests
from datetime import datetime
from random import random
import uuid
import math

from config import SUPERVISOR_SYSTEM_PROMPTV4
from utils import euclidean_distance
from typing import TypedDict, Annotated, Literal, List

import streamlit as st
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, AnyMessage, ToolMessage
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

# ========================= TOOLS =========================
def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Tính khoảng cách (km) giữa 2 toạ độ bằng công thức Haversine."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


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
        user_lat, user_lon = 21.0278, 105.8342

    salons: List[dict] = [
        {"id": 1, "name": "Salon Hoa Mai", "address": "Số 10, Quận Hoàn Kiếm, Hà Nội", "lat": 21.0290, "lon": 105.8536, "rating": 4.6},
        {"id": 2, "name": "Hair Studio 88", "address": "Khu vực Ba Đình, Hà Nội", "lat": 21.0340, "lon": 105.8285, "rating": 4.4},
        {"id": 3, "name": "Tóc & Spa Tây Hồ", "address": "Tây Hồ, Hà Nội", "lat": 21.0630, "lon": 105.8210, "rating": 4.2},
        {"id": 4, "name": "Salon Minh Châu", "address": "Hai Bà Trưng, Hà Nội", "lat": 21.0125, "lon": 105.8532, "rating": 4.1},
        {"id": 5, "name": "Barber Street", "address": "Cầu Giấy, Hà Nội", "lat": 21.0295, "lon": 105.7836, "rating": 4.3},
        {"id": 6, "name": "Salon Gội Đầu Thư Giãn", "address": "Đống Đa, Hà Nội", "lat": 21.0115, "lon": 105.8467, "rating": 4.0},
    ]

    results = []
    for s in salons:
        dist_km = _haversine(user_lat, user_lon, s["lat"], s["lon"])
        s_copy = s.copy()
        s_copy["distance_km"] = round(dist_km, 3)
        results.append(s_copy)

    max_radius_km = 8.0
    nearby = [r for r in results if r["distance_km"] <= max_radius_km]
    nearby.sort(key=lambda x: x["distance_km"])

    if not nearby:
        results.sort(key=lambda x: x["distance_km"])
        nearby = results[:3]

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


from langchain_core.utils.function_calling import convert_to_openai_tool

# ========================= LLM =========================
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

# Danh sách tool
booking_tools = [get_near_salon, check_availability, book_appointment, cancel_appointment, list_branches]


class BookingState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    plan: dict  # Lưu plan hiện tại từ planner
    pending_tools: List[str]  # ✨ V6: Danh sách tools đang chờ execute


# ========================= V6 SCHEMAS =========================
# ✨ V6: ToolCall chỉ có tool_name, KHÔNG có tool_args
class ToolSelection(BaseModel):
    tool_name: str = Field(description="Tên tool cần gọi (phải chính xác)")
    # reason: str = Field(default="", description="Lý do ngắn gọn tại sao chọn tool này")


# ✨ V6: BookingPlan hỗ trợ NHIỀU tools (selected_tools là List)
class BookingPlan(BaseModel):
    thought: str = Field(description="Phân tích đã có gì thiếu gì và cần làm gì tiếp theo (tối đa 2 câu)")
    next_action: Literal["call_tool", "ask_user", "respond"] = Field(
        description="Hành động tiếp theo: call_tool (gọi tool), ask_user (hỏi thêm), respond (trả lời)"
    )
    selected_tools: List[ToolSelection] = Field(
        default=[],
        description="Danh sách tools cần gọi (có thể chọn nhiều tools để gọi song song). Chỉ điền khi next_action=call_tool"
    )
    response: str = Field(default="", description="Câu trả lời cho user (nếu next_action=ask_user hoặc respond)")


# Hàm an toàn để lấy tên + tham số + description (giữ nguyên từ V5)
def safe_tool_description(tool_func):
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
            param_parts = []
            for pname, pinfo in props.items():
                ptype = pinfo.get('type', 'any')
                param_parts.append(f"{pname}: {ptype}")
            param_str = ", ".join(param_parts)

        return f"• {name}({param_str}) → {desc}"
    except Exception as e:
        print(f"Tool desc error for {tool_name}: {e}")
        return f"• {tool_name}(params unknown) → Tool for {tool_name}"


TOOL_DESCRIPTIONS = "\n".join([safe_tool_description(t) for t in booking_tools])
print("TOOL_DESCRIPTIONS generated:", TOOL_DESCRIPTIONS)


# ========================= V6 BOOKING PLANNER =========================
def booking_planner(state: BookingState):
    """
    ✨ V6 Planner - CHỈ chọn tools, KHÔNG extract arguments
    Output: BookingPlan với thought, next_action, selected_tools (chỉ có tool_name), response
    """
    messages = state["messages"]

    system_prompt = f"""
Today is: {datetime.now().strftime("%d/%m/%Y %H:%M")}

Bạn là Janie – trợ lý đặt lịch 30Shine thông minh. Nhiệm vụ: phân tích và lên kế hoạch. Sử dụng linh hoạt các tool có sẵn

=== TOOLS CÓ SẴN ===
{TOOL_DESCRIPTIONS}

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

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{messages}")
    ])

    print(messages)

    chain = prompt | llm.with_structured_output(BookingPlan, method="function_calling")

    try:
        plan: BookingPlan = chain.invoke({"messages": messages})
        print(f"\n[PLANNER] 💭 Thought: {plan.thought}")
        print(f"[PLANNER] ➡️ Next Action: {plan.next_action}")
        if plan.selected_tools:
            print(f"[PLANNER] 🔧 Selected Tools ({len(plan.selected_tools)}):")
            for t in plan.selected_tools:
                print(f"   - {t.tool_name}")
        if plan.response:
            print(f"[PLANNER] 💬 Response: {plan.response[:100]}...")
    except Exception as e:
        print(f"[PLANNER] ❌ Error: {e}")
        plan = BookingPlan(
            thought="Lỗi parse, cần hỏi lại",
            next_action="ask_user",
            response="Dạ anh ơi, em chưa hiểu lắm, anh nói rõ hơn được không ạ?"
        )

    # ✨ V6: Lưu danh sách tools cần execute
    pending_tools = [t.tool_name for t in plan.selected_tools] if plan.next_action == "call_tool" else []

    return {
        "plan": plan.model_dump(),
        "pending_tools": pending_tools
    }


# ========================= V6 TOOL EXECUTOR (Function Calling) =========================
def tool_executor(state: BookingState):
    """
    ✨ V6 Tool Executor:
    - Nhận danh sách tools từ planner (chỉ có tool_name)
    - Gọi ChatGPT với function calling để extract arguments
    - Execute tools và trả về kết quả
    """
    pending_tools = state.get("pending_tools", [])
    messages = state["messages"]

    print(f"\n[TOOL_EXECUTOR] 🔧 Tools cần execute: {pending_tools}")

    if not pending_tools:
        print("[TOOL_EXECUTOR] ⚠️ Không có tools nào để execute")
        return {"pending_tools": []}

    # Lọc tools được chọn
    tool_map = {t.name: t for t in booking_tools}
    selected_tools = [tool_map[name] for name in pending_tools if name in tool_map]

    if not selected_tools:
        print("[TOOL_EXECUTOR] ❌ Không tìm thấy tools")
        return {"pending_tools": []}

    # ✨ V6: Gọi ChatGPT với function calling để extract arguments
    system_prompt = f"""
Bạn là assistant trích xuất thông tin từ cuộc hội thoại để gọi tools.
Hãy phân tích cuộc hội thoại và gọi các tools phù hợp với thông tin có sẵn.

Today is: {datetime.now().strftime("%d/%m/%Y %H:%M")}

QUAN TRỌNG:
- Bạn PHẢI gọi các tools sau: {pending_tools}
- Nếu thiếu thông tin bắt buộc, hãy dùng giá trị mặc định hợp lý hoặc suy luận từ context
- Với ngày/giờ: nếu user nói "mai", "chiều nay", etc. → convert sang format cụ thể
- Với địa chỉ: lấy từ context hoặc dùng thông tin user đã cung cấp
"""

    llm_with_tools = ChatOpenAI(model="gpt-4.1-mini", temperature=0).bind_tools(
        selected_tools,
        tool_choice="required"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{messages}")
    ])

    try:
        response = (prompt | llm_with_tools).invoke({"messages": messages})
        print(f"[TOOL_EXECUTOR] 📨 LLM Response received")

        new_messages = []

        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"[TOOL_EXECUTOR] 🎯 Có {len(response.tool_calls)} tool calls")

            # Thêm AIMessage với tool_calls
            new_messages.append(response)

            for tc in response.tool_calls:
                tool_name = tc['name']
                tool_args = tc['args']
                tool_call_id = tc['id']

                print(f"[TOOL_EXECUTOR] 🔨 Executing: {tool_name}({tool_args})")

                if tool_name in tool_map:
                    try:
                        result = tool_map[tool_name].invoke(tool_args)
                        print(f"[TOOL_EXECUTOR] ✅ Result: {str(result)[:200]}...")
                    except Exception as e:
                        result = f"Lỗi khi gọi {tool_name}: {e}"
                        print(f"[TOOL_EXECUTOR] ❌ Error: {result}")
                else:
                    result = f"Tool {tool_name} không tồn tại"

                # Thêm ToolMessage
                new_messages.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call_id,
                    name=tool_name
                ))
        else:
            print("[TOOL_EXECUTOR] ⚠️ LLM không gọi tool nào")

    except Exception as e:
        print(f"[TOOL_EXECUTOR] ❌ Exception: {e}")
        new_messages = []

    return {
        "messages": new_messages,
        "pending_tools": []
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
            ("system",
             "Tổng hợp lại cuộc trò chuyện thành câu trả lời tự nhiên. Xưng em, gọi anh, kết thúc bằng ạ."),
            ("placeholder", "{messages}")
        ])
        text = (prompt | llm).invoke({"messages": state["messages"]}).content

    print(f"[BOOKING_RESPONDER] 📝 Final: {text}")
    print("="*50)
    return {"messages": [AIMessage(content=text, name="booking_node")]}


def booking_router(state: BookingState):
    """
    Router dựa trên plan.next_action:
    - "call_tool" → tool_executor
    - "ask_user" / "respond" → responder
    """
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
    """
    Graph structure:

    START → planner → [router] → tool_executor → planner (loop)
                         ↓
                      respond → END
    """
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


# ========================= INFORMATION AGENT =========================
info_tools = [get_info, list_branches, get_near_salon]


class InfoPlan(BaseModel):
    thought: str = Field(description="Suy nghĩ NGẮN GỌN (1 câu) về hành động cần làm")
    next_action: Literal["call_tool", "respond"] = Field(description="Hành động tiếp theo")
    selected_tools: List[ToolSelection] = Field(default=[], description="Tools cần gọi nếu next_action=call_tool")
    response: str = Field(default="", description="Câu trả lời nếu next_action=respond")


def info_planner(state):
    print("\n" + "="*50)
    print("[INFO_PLANNER] ▶ Planning")

    info_tool_desc = "\n".join([safe_tool_description(t) for t in info_tools])

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""Bạn là Janie chuyên tư vấn dịch vụ 30Shine.

=== TOOLS ===
{info_tool_desc}

Nếu biết câu trả lời → next_action="respond", điền response
Nếu cần tra cứu → next_action="call_tool", điền selected_tools (CHỈ CẦN tool_name, KHÔNG cần arguments)

⚡ QUAN TRỌNG:
- thought: CHỈ 1 câu ngắn (vd: "Cần lấy thông tin giá")

Phong cách: xưng em, gọi anh, kết thúc bằng ạ."""),
        ("placeholder", "{messages}")
    ])

    try:
        plan = (prompt | llm.with_structured_output(InfoPlan, method="function_calling")).invoke({"messages": state["messages"]})
        print(f"[INFO_PLANNER] 💭 {plan.thought}")
        print(f"[INFO_PLANNER] ➡️ {plan.next_action}")
        if plan.selected_tools:
            print(f"[INFO_PLANNER] 🔧 Tools: {[t.tool_name for t in plan.selected_tools]}")
    except Exception as e:
        print(f"[INFO_PLANNER] ❌ Error: {e}")
        plan = InfoPlan(thought="Lỗi", next_action="respond", response="Dạ anh ơi, em chưa hiểu, anh hỏi lại nhé ạ!")

    pending_tools = [t.tool_name for t in plan.selected_tools] if plan.next_action == "call_tool" else []

    return {
        "messages": [AIMessage(content=f"[Info Plan] {plan.thought}", name="info_planner")],
        "plan": plan.model_dump(),
        "pending_tools": pending_tools
    }


def info_tool_executor(state):
    """✨ V6: Dùng function calling để extract args"""
    pending_tools = state.get("pending_tools", [])
    messages = state["messages"]

    print(f"[INFO_TOOL] 🔧 Tools: {pending_tools}")

    if not pending_tools:
        return {"pending_tools": []}

    tool_map = {t.name: t for t in info_tools}
    selected_tools = [tool_map[name] for name in pending_tools if name in tool_map]

    if not selected_tools:
        return {"pending_tools": []}

    llm_with_tools = ChatOpenAI(model="gpt-4.1-mini", temperature=0).bind_tools(
        selected_tools,
        tool_choice="required"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"Trích xuất thông tin và gọi tools: {pending_tools}. Today: {datetime.now().strftime('%d/%m/%Y')}"),
        ("placeholder", "{messages}")
    ])

    try:
        response = (prompt | llm_with_tools).invoke({"messages": messages})
        new_messages = []

        if hasattr(response, 'tool_calls') and response.tool_calls:
            new_messages.append(response)

            for tc in response.tool_calls:
                tool_name = tc['name']
                tool_args = tc['args']
                tool_call_id = tc['id']

                print(f"[INFO_TOOL] 🔨 {tool_name}({tool_args})")

                if tool_name in tool_map:
                    try:
                        result = tool_map[tool_name].invoke(tool_args)
                    except Exception as e:
                        result = f"Lỗi: {e}"
                else:
                    result = f"Tool {tool_name} không tồn tại"

                new_messages.append(ToolMessage(content=str(result), tool_call_id=tool_call_id, name=tool_name))

    except Exception as e:
        print(f"[INFO_TOOL] ❌ Error: {e}")
        new_messages = []

    return {
        "messages": new_messages,
        "pending_tools": []
    }


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


# ========================= SUPERVISOR GRAPH =========================
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    query: str
    original_query: str
    chat_history: Annotated[list[AnyMessage], add_messages]
    list_tasks: List[dict]


class Action(BaseModel):
    name: Literal["booking_node", "information_node"]
    query: str = Field(..., description="Query gửi cho agent này")


class SupervisorPlan(BaseModel):
    actions: List[Action]


booking_agent = create_booking_agent()
information_agent = create_information_agent()


def group_tasks_by_agent(actions: List[Action], original_query: str) -> List[dict]:
    """Group multiple tasks for same agent into 1 task with original query"""
    seen_agents = set()
    grouped_tasks = []

    for action in actions:
        if action.name not in seen_agents:
            seen_agents.add(action.name)
            grouped_tasks.append({
                "name": action.name,
                "query": original_query,
                "status": "pending"
            })

    return grouped_tasks


def supervisor_node(state: AgentState):
    print("\n" + "#"*60)
    print("[SUPERVISOR] 🎯 Supervisor đang xử lý...")
    print(f"[SUPERVISOR] Current tasks: {state.get('list_tasks', [])}")

    for task in state.get("list_tasks", []):
        if task.get("status") != "done":
            print(f"[SUPERVISOR] → Tiếp tục task chưa hoàn thành: {task['name']}")
            print("#"*60)
            return Command(goto=task["name"], update={"query": task["query"]})

    if state.get("list_tasks") and all(t.get("status") == "done" for t in state["list_tasks"]):
        print("[SUPERVISOR] ✅ Tất cả tasks hoàn thành → END")
        print("#"*60)
        return Command(goto=END)

    print("[SUPERVISOR] 🔍 Phân tích query mới...")
    messages = [{"role": "system", "content": SUPERVISOR_SYSTEM_PROMPTV4}] + state["messages"]
    response = llm.with_structured_output(SupervisorPlan).invoke(messages)
    print(f"SUPERVISOR RESPONSE: {response}")

    original_query = state.get("original_query") or state.get("query", "")
    tasks = group_tasks_by_agent(response.actions, original_query)

    print(f"[SUPERVISOR] 📋 Raw actions: {[a.name for a in response.actions]}")
    print(f"[SUPERVISOR] ✨ Grouped tasks: {[t['name'] for t in tasks]}")

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

    result = booking_agent.invoke({
        "messages": state["chat_history"] + [HumanMessage(content=state["query"])],
        "pending_tools": []
    })
    final_msg = result["messages"][-1].content

    print(f"[BOOKING_NODE] 📝 Kết quả: {final_msg[:100]}...")

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

    result = information_agent.invoke({
        "messages": state["chat_history"] + [HumanMessage(content=state["query"])],
        "pending_tools": []
    })
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
st.set_page_config(page_title="30Shine Agent", layout="wide")
st.title("30Shine Multi-Agent System V6")
st.caption("✨ NEW: Planner chỉ chọn tools (không extract args) + Executor dùng Function Calling | Parallel Tools Support")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "30shine_chat_001"

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

if prompt := st.chat_input("Anh ơi, em nghe nè ạ..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.spinner("Janie đang suy nghĩ..."):
        config = {"configurable": {"thread_id": st.session_state.thread_id}}

        inputs = {
            "messages": [HumanMessage(content=prompt)],
            "query": prompt,
            "original_query": prompt,
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
