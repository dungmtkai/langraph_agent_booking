import json
import requests
from datetime import datetime
from random import random
from utils import euclidean_distance
from typing import TypedDict, Annotated, Literal, List

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
    # … thêm nếu cần
}

# ========================= TOOLS =========================
@tool
def get_near_salon(user_address: str, city: str = "Hà Nội") -> str:
    """Tìm và gợi ý salon 30Shine gần nhất dựa trên địa chỉ và thành phố của khách."""
    url = f"https://geocode.search.hereapi.com/v1/geocode?q={user_address}+{city}&apiKey=A7V_JCsxV2Y_A_WBg00q_mUB-bDCynwEhwaZeT6QfwY&limit=1"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = json.loads(response.content.decode('utf-8-sig'))
        res = data["items"][0]
        city_id = CITY_IDS.get(res['address']['county'])
        if city_id is None:
            return "Không tìm thấy thành phố phù hợp. Vui lòng thử lại."

        lat_lon = res['position']
        near_salon = {'city_id': city_id, 'lat': lat_lon['lat'], 'lon': lat_lon['lng']}

        url_get_all_salon = "https://storage.30shine.com/web/v3/configs/get_all_salon.json"
        response = requests.get(url_get_all_salon, timeout=5)
        response.raise_for_status()
        data = json.loads(response.content.decode('utf-8-sig'))

        salons = [x for x in data["data"] if x["cityId"] == near_salon['city_id']]
        salons.sort(
            key=lambda x: euclidean_distance(
                near_salon['lat'], near_salon['lon'], x['latitude'], x['longitude']
            )
        )

        if not salons:
            return "Không tìm thấy salon nào gần khu vực của bạn."

        list_salon = "Danh sách salon\n" + "\n".join(
            f"- **{x['addressNew']}**" for x in salons[:5]
        )
        return list_salon
    except (requests.RequestException, json.JSONDecodeError, KeyError):
        return "Dạ xin lỗi, em không thể cung cấp thông tin này."

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

from langchain_core.utils.function_calling import convert_to_openai_tool
# Đầu tiên: chuyển tất cả tool thành định dạng OpenAI tool spec
booking_tool_specs = [
    convert_to_openai_tool(t) for t in [
        get_near_salon,
        check_availability,
        book_appointment,
        cancel_appointment,
        list_branches,
    ]
]

# Tạo chuỗi mô tả tool đẹp để nhét vào prompt
TOOLS_DESCRIPTION = "\n".join([
    f"- {tool['function']['name']}: {tool['function']['description']}"
    for tool in booking_tool_specs
])

# Prompt cuối cùng – LLM biết rõ mình có gì
BOOKING_SYSTEM_PROMPT_WITH_TOOLS = f"""
Today is: {{date_time}}

Bạn là Janie – trợ lý đặt lịch siêu thông minh của 30Shine.
Bạn chỉ được phép gọi tool nếu cần thiết, và chỉ gọi đúng 1 tool mỗi lần.

=== DANH SÁCH TOOL BẠN CÓ THỂ DÙNG ===
{TOOLS_DESCRIPTION}

QUY TẮC BẮT BUỘC:
- Nếu thiếu thông tin (địa chỉ, ngày, giờ, số điện thoại) → để vào missing_info, KHÔNG gọi tool
- Nếu cần tìm salon → gọi get_near_salon
- Nếu cần kiểm tra slot → gọi check_availability (phải có salon_address)
- Nếu đủ thông tin và khách xác nhận → gọi book_appointment
- Nếu khách muốn hủy → gọi cancel_appointment
- Chỉ trả về đúng schema BookingPlan, không thêm text thừa.

Phong cách: xưng em, gọi anh, kết thúc bằng “ạ”.
"""

SUPERVISOR_SYSTEM_PROMPT = """
Bạn là Supervisor điều phối 2 agent chuyên biệt:

booking_node → Đặt lịch, hủy lịch, check slot, tìm salon gần nhất
information_node → Tư vấn giá, dịch vụ, combo, tiện ích, chỗ để xe, v.v.

Phân tích query và trả về đúng JSON theo schema.
"""




# ========================= CUSTOM BOOKING AGENT (thông minh + sequential) =========================
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

# Danh sách tool
booking_tools = [get_near_salon, check_availability, book_appointment, cancel_appointment, list_branches]


class BookingState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    plan: dict  # Lưu plan hiện tại từ planner


# Schema cho Plan - LLM sẽ output theo format này
class ToolCall(BaseModel):
    tool_name: str = Field(description="Tên tool cần gọi (phải chính xác)")
    tool_args: dict = Field(default={}, description="Arguments cho tool")


class BookingPlan(BaseModel):
    thought: str = Field(description="Suy nghĩ chi tiết bằng tiếng Việt về tình huống hiện tại")
    analysis: str = Field(description="Phân tích: đã có gì, còn thiếu gì")
    next_action: Literal["call_tool", "ask_user", "respond"] = Field(
        description="Hành động tiếp theo: call_tool (gọi tool), ask_user (hỏi thêm), respond (trả lời)"
    )
    tool_call: ToolCall | None = Field(default=None, description="Thông tin tool cần gọi (nếu next_action=call_tool)")
    response: str = Field(default="", description="Câu trả lời cho user (nếu next_action=ask_user hoặc respond)")


# Hàm an toàn để lấy tên + tham số + description
def safe_tool_description(tool_func):
    # Lấy tên tool một cách an toàn (hỗ trợ cả function và StructuredTool)
    tool_name = getattr(tool_func, 'name', None) or getattr(tool_func, '__name__', str(tool_func))

    try:
        spec = convert_to_openai_tool(tool_func)
        func = spec.get('function', {})
        name = func.get('name', tool_name)
        desc = func.get('description', 'No description')

        # Safe param extraction
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


# Generate TOOL_DESCRIPTIONS (gọi 1 lần ngoài hàm)
# booking_tools = [get_near_salon, check_availability, book_appointment, cancel_appointment, list_branches]
TOOL_DESCRIPTIONS = "\n".join([safe_tool_description(t) for t in booking_tools])
print("TOOL_DESCRIPTIONS generated:", TOOL_DESCRIPTIONS)  # Để debug


def booking_planner(state: BookingState):
    """
    Planner thông minh - suy nghĩ và lên plan trước khi hành động.
    Output: BookingPlan với thought, analysis, next_action, tool_call, response
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
   - "call_tool": Nếu đủ thông tin để gọi tool → điền tool_call
   - "ask_user": Nếu thiếu thông tin quan trọng → điền response (câu hỏi)
   - "respond": Nếu đã hoàn thành hoặc chỉ cần trả lời → điền response

=== QUY TẮC QUAN TRỌNG ===
- Nếu có địa chỉ (tên đường + thành phố) → GỌI get_near_salon ngay, KHÔNG hỏi thêm
- Nếu khách muốn đặt lịch nhưng chưa có salon → tìm salon trước
- Nếu có salon + ngày giờ → check_availability
- Nếu slot OK + có SĐT → book_appointment
- Phong cách response: xưng em, gọi anh, kết thúc bằng "ạ"

=== QUAN TRỌNG: CÁCH ĐIỀN tool_call ===
Khi next_action="call_tool", BẮT BUỘC phải điền đầy đủ tool_call:
- tool_name: tên tool chính xác (get_near_salon, check_availability, book_appointment, cancel_appointment, list_branches)
- tool_args: dict chứa các tham số cần thiết, PHẢI ĐIỀN ĐẦY ĐỦ

Ví dụ tool_args cho từng tool:
- get_near_salon: tool_args phải có "user_address" (bắt buộc), "city" (optional, default "Hà Nội")
- check_availability: tool_args phải có "salon_address", "date", "time"
- book_appointment: tool_args phải có "salon_address", "date", "time", "phone"
- cancel_appointment: tool_args phải có "phone"
- list_branches: không cần tool_args
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{messages}")
    ])

    # Dùng method="function_calling" vì OpenAI structured output không hỗ trợ dict type
    chain = prompt | llm.with_structured_output(BookingPlan, method="function_calling")

    try:
        plan: BookingPlan = chain.invoke({"messages": messages})
        print(f"\n[PLANNER] 💭 Thought: {plan.thought}")
        print(f"[PLANNER] 📊 Analysis: {plan.analysis}")
        print(f"[PLANNER] ➡️ Next Action: {plan.next_action}")
        if plan.tool_call:
            print(f"[PLANNER] 🔧 Tool: {plan.tool_call.tool_name}({plan.tool_call.tool_args})")
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
    Thực thi tool dựa trên plan từ planner.
    Dùng HumanMessage thay vì ToolMessage để tránh lỗi OpenAI validation.
    """
    plan = state.get("plan", {})
    tool_call = plan.get("tool_call") or {}
    tool_name = tool_call.get("tool_name", "")
    tool_args = tool_call.get("tool_args") or {}

    print(f"\n[TOOL_EXECUTOR] 🔧 Executing: {tool_name}")
    print(f"[TOOL_EXECUTOR] 📥 Args: {tool_args}")

    # Tìm và gọi tool
    tool_map = {t.name: t for t in booking_tools}

    if tool_name not in tool_map:
        result = f"Tool '{tool_name}' không tồn tại. Tools có sẵn: {list(tool_map.keys())}"
        print(f"[TOOL_EXECUTOR] ❌ {result}")
    elif not tool_args:
        # Nếu không có args, báo lỗi để planner thử lại
        result = f"Thiếu arguments cho tool {tool_name}. Tool này cần các tham số."
        print(f"[TOOL_EXECUTOR] ⚠️ {result}")
    else:
        try:
            tool = tool_map[tool_name]
            result = tool.invoke(tool_args)
            print(f"[TOOL_EXECUTOR] ✅ Result: {result}")
        except Exception as e:
            result = f"Lỗi khi gọi {tool_name}: {e}"
            print(f"[TOOL_EXECUTOR] ❌ {result}")

    # Dùng HumanMessage với prefix [Tool Result] thay vì ToolMessage
    # Vì OpenAI không chấp nhận ToolMessage nếu không có tool_calls trước đó
    return {
        "messages": [HumanMessage(content=f"[Tool Result - {tool_name}]: {result}")]
    }


def booking_responder(state: BookingState):
    """
    Tạo response cuối cùng từ plan.response
    """
    print("\n" + "="*50)
    print("[BOOKING_RESPONDER] ▶ Tạo response")

    plan = state.get("plan", {})
    response = plan.get("response", "")

    if response:
        text = response
        print(f"[BOOKING_RESPONDER] ✅ Dùng response từ plan: {text[:100]}...")
    else:
        # Fallback: tổng hợp từ conversation
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

    # Nodes
    g.add_node("planner", booking_planner)
    g.add_node("tool_executor", tool_executor)
    g.add_node("respond", booking_responder)

    # Edges
    g.add_edge(START, "planner")
    g.add_conditional_edges(
        "planner",
        booking_router,
        {"tool_executor": "tool_executor", "respond": "respond"}
    )
    g.add_edge("tool_executor", "planner")  # Loop back sau khi có tool result
    g.add_edge("respond", END)

    return g.compile(checkpointer=MemorySaver())


# ========================= CUSTOM INFORMATION AGENT =========================
info_tools = [get_info, list_branches, get_near_salon]


class InfoPlan(BaseModel):
    thought: str = Field(description="Suy nghĩ về câu hỏi của khách")
    next_action: Literal["call_tool", "respond"] = Field(description="Hành động tiếp theo")
    tool_call: ToolCall | None = Field(default=None, description="Tool cần gọi nếu next_action=call_tool")
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
Nếu cần tra cứu → next_action="call_tool", điền tool_call

Phong cách: xưng em, gọi anh, kết thúc bằng ạ."""),
        ("placeholder", "{messages}")
    ])

    try:
        plan = (prompt | llm.with_structured_output(InfoPlan, method="function_calling")).invoke({"messages": state["messages"]})
        print(f"[INFO_PLANNER] 💭 {plan.thought}")
        print(f"[INFO_PLANNER] ➡️ {plan.next_action}")
    except Exception as e:
        print(f"[INFO_PLANNER] ❌ Error: {e}")
        plan = InfoPlan(thought="Lỗi", next_action="respond", response="Dạ anh ơi, em chưa hiểu, anh hỏi lại nhé ạ!")

    return {
        "messages": [AIMessage(content=f"[Info Plan] {plan.thought}", name="info_planner")],
        "plan": plan.model_dump()
    }


def info_tool_executor(state):
    plan = state.get("plan", {})
    tool_call = plan.get("tool_call") or {}
    tool_name = tool_call.get("tool_name", "")
    tool_args = tool_call.get("tool_args") or {}

    print(f"[INFO_TOOL] 🔧 Executing: {tool_name}")

    tool_map = {t.name: t for t in info_tools}
    if tool_name not in tool_map:
        result = f"Tool '{tool_name}' không tồn tại"
    elif not tool_args and tool_name not in ["get_info", "list_branches"]:  # Một số tool không cần args
        result = f"Thiếu arguments cho tool {tool_name}"
    else:
        try:
            result = tool_map[tool_name].invoke(tool_args)
        except Exception as e:
            result = f"Lỗi: {e}"

    print(f"[INFO_TOOL] ✅ Result: {result}")
    return {"messages": [HumanMessage(content=f"[Tool Result - {tool_name}]: {result}")]}


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
    g = StateGraph(BookingState)  # Reuse BookingState (có plan)
    g.add_node("planner", info_planner)
    g.add_node("tool_executor", info_tool_executor)
    g.add_node("respond", info_responder)

    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", info_router, {"tool_executor": "tool_executor", "respond": "respond"})
    g.add_edge("tool_executor", "planner")
    g.add_edge("respond", END)

    return g.compile(checkpointer=MemorySaver())


# ========================= SUPERVISOR GRAPH (không validator) =========================
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    query: str
    chat_history: Annotated[list[AnyMessage], add_messages]
    list_tasks: List[dict]  # [{"name": "booking_node", "status": "done"}]


class Action(BaseModel):
    name: Literal["booking_node", "information_node"]
    query: str = Field(..., description="Query gửi cho agent này")


class AgentRequest(BaseModel):
    thought: str = Field(description="Your reasoning about what to do next")
    action: List[Action] = Field(description="List of actions with agent names and their queries")


class SupervisorPlan(BaseModel):
    actions: List[Action]


booking_agent = create_booking_agent()
information_agent = create_information_agent()


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
    tasks = [{"name": a.name, "query": a.query, "status": "pending"} for a in response.actions]

    print(f"[SUPERVISOR] 📋 Tasks được tạo: {[t['name'] for t in tasks]}")
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
    print(f"[BOOKING_NODE] Chat history length: {len(state.get('chat_history', []))}")

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
    print(f"[INFO_NODE] Chat history length: {len(state.get('chat_history', []))}")

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
st.set_page_config(page_title="30Shine Agent", layout="wide")
st.title("30Shine Multi-Agent System")
st.caption("Custom Graph + Sequential Tool Calling – Không Validator")

# KHỞI TẠO SESSION STATE (BẮT BUỘC!)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "30shine_chat_001"

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

# Input từ user
if prompt := st.chat_input("Anh ơi, em nghe nè ạ..."):
    # Lưu tin nhắn user
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.spinner("Janie đang suy nghĩ..."):
        config = {"configurable": {"thread_id": st.session_state.thread_id}}

        inputs = {
            "messages": [HumanMessage(content=prompt)],
            "query": prompt,
            "chat_history": st.session_state.chat_history,
            "list_tasks": []
        }

        final_answer = None
        try:
            for chunk in app.stream(inputs, config=config, stream_mode="values"):
                msgs = chunk.get("messages", [])
                if msgs and msgs[-1].content:
                    last_msg = msgs[-1]
                    # Chỉ lấy tin nhắn từ agent thực sự kiện
                    if hasattr(last_msg, "name") and last_msg.name in ["booking_node", "information_node"]:
                        final_answer = last_msg.content
                    # Trường hợp supervisor trả lời trực tiếp (nếu có)
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