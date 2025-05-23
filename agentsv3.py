from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

memory = MemorySaver()

from config import BOOKING_SYSTEM_PROMPT, SUPERVISOR_SYSTEM_PROMPTV3, valid_system_prompt, SUPERVISOR_SYSTEM_PROMPTV4
from tools import book_appointment, cancel_appointment, check_availability, get_near_salon, list_branches, \
    get_info
from dotenv import load_dotenv
from langgraph.graph import END
from typing import Annotated, TypedDict, Literal, Any, List
from langgraph.graph import StateGraph, START
from langgraph.types import Command
from datetime import datetime

load_dotenv()
from langgraph.graph.message import add_messages

from pydantic import BaseModel, Field

next_des = ("Determines which specialist to activate next in the workflow sequence:"
            "'booking_node' Hỗ trợ khách hàng trong việc đặt lịch hoặc thay đổi lịch hẹn (không bao gồm email hoặc tên), kiểm tra các khung giờ còn trống tại salon, tìm salon gần nhất và hiển thị các chi nhánh salon. Giao các nhiệm vụ liên quan đến đặt lịch cho trợ lý này."
            "'information_node' Cung cấp thông tin tư vấn chi tiết cho khách hàng về các dịch vụ của 30Shine, bảng giá, nhân viên, so sánh giữa các salon, gói combo, tiện ích và chỗ đỗ xe tại cả salon thường và salon cao cấp. Giao các nhiệm vụ liên quan đến câu hỏi thường gặp (FAQ) cho trợ lý này."
            )


class AgentState(TypedDict):
    messages: Annotated[list[Any], add_messages]
    query: str
    chat_history: Annotated[list[Any], add_messages]
    list_tasks: List


# class Router(BaseModel):
#     next: Literal["information_node", "booking_node", "fallback_node"] = Field(description=next_des)
#     reason: str = Field(
#         description="Detailed justification for the routing decision, explaining the rationale behind selecting the particular specialist and how this advances the task toward completion."
#     )
#     input: str = Field(description="Yêu cầu được giao cho agent phù hợp với moo tả về công việc của agent")

class TaskItem(BaseModel):
    agent: str = Field(..., description="Tên của agent được giao xử lý task (ví dụ: 'fallback_node', 'booking_node', 'information_node')")
    task: str = Field(..., description="Nhiệm vụ cụ thể được trích xuất từ truy vấn người dùng, dành cho agent xử lý")


class AgentResponse(BaseModel):
    response: List[TaskItem] = Field(..., description="Danh sách các task đã được phân tích và phân công cho từng agent")
current_date = datetime.now()

openai_model = ChatOpenAI(model="gpt-4o-mini")

members_dict = {
    'booking_node': 'Hỗ trợ khách hàng trong việc đặt lịch hoặc thay đổi lịch hẹn (không bao gồm email hoặc tên), kiểm tra các khung giờ còn trống tại salon, tìm salon gần nhất và hiển thị các chi nhánh salon. Giao các nhiệm vụ liên quan đến đặt lịch cho trợ lý này.',
    'information_node': 'Hỗ trợ tư vấn chi tiết cho khách hàng về các dịch vụ của 30Shine, bảng giá, nhân viên, so sánh giữa các salon, gói combo, tiện ích và chỗ đỗ xe tại cả salon thường và salon cao cấp. Giao các nhiệm vụ liên quan đến câu hỏi thường gặp (FAQ) cho trợ lý này.',
    'fallback_agent': "Trả lời các câu hỏi không ằm trong phạm vi của các agent khác, trả lời các câu hỏi về nhiệm vụ bạn có thể làm gi"}

worker_info = '\n\n'.join([f'WORKER: {member} \nDESCRIPTION: {description}' for member, description in
                           members_dict.items()])


def booking_node(state: AgentState) -> Command[Literal['validator']]:
    print("*****************called booking node************")

    system_prompt = BOOKING_SYSTEM_PROMPT.format(date_time=current_date)

    system_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt
            ),
            (
                "placeholder",
                "{messages}"
            ),
        ]
    )
    booking_agent = create_react_agent(model=openai_model,
                                       tools=[book_appointment],
                                       version="v2",
                                       prompt=system_prompt, debug=True)

    query_part = [
        HumanMessage(content=state["query"])
    ]

    input_for_agent = {"messages": state["chat_history"] + query_part}
    print(f"______________input for booking_node_______________:{input_for_agent}")
    result = booking_agent.invoke(input_for_agent)
    for task in state["list_tasks"]:
        if task["agent"] == "booking_node":
            task["status"] = "done"
            break

    return Command(
        update={
            "messages": [
                AIMessage(
                    content=result["messages"][-1].content,
                    name="booking_node"
                )
            ]
        },
        goto="supervisor",
    )


def fallback_node(state: AgentState) -> Command[Literal['validator']]:
    print("*****************called fallback node************")

    system_prompt = BOOKING_SYSTEM_PROMPT.format(date_time=current_date)

    system_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Bạn là Janie, trợ lý ảo tư vấn và chăm sóc khách hàng của 30Shine, một chuỗi salon và hairdressing dành cho nam giới và trẻ nhỏ.
Bạn có khả năng:
- Hỗ trợ khách hàng đặt lịch/sửa lịch (đặt lịch dựa trên số điện thoại, không dựa theo email, tên tuổi)
- Kiểm tra khung giờ còn trống tại các salon.
- Cung cấp danh sách các salon theo hạng (thường hoặc thương gia).
- Tìm địa chỉ hoặc gợi ý salon gần nhất dựa trên vị trí khách hàng cung cấp.
- Cung cấp các thông tin chi nhánh, cơ sở, salon của 30shine
- Tư vấn khách hàng về sản phẩm, dịch vụ, giá cả, chương trình ưu đãi, và combo liên quan.
- Cung cấp thông tin về nhân viên làm việc tại salon là nam hay nữ
- Thông tin, so sánh các salon, thời gian phục vụ và các tiện ích khác
- Tư vấn các dịch vụ cắt tóc, gội đầu, nhuộm tóc, chăm sóc tóc, massage cổ vai gáy và full body, chăm sóc da mặt, lấy ráy tay, kiểu tóc, tình trạng tóc...và combo liên quan đến cả 2 salon thường và salon thương gia của 30Shine. Nếu khách hỏi combo cùng có ở cả 2 salon thường và salon thương gia thì đưa ra cả 2 cho khách lựa chọn
- Tư vấn cung cấp các thông tin liên quan đến chỗ đỗ xe, chỗ để xe ô tô, bị phạt hay an toàn ...
- Tư vấn thông tin về bãi đỗ xe liên quan tới các địa chỉ tại salon 30 Shine.
Hãy trả lời câu hỏi của người dùng
"""

            ),
            (
                "placeholder",
                "{messages}"
            ),
        ]
    )
    booking_agent = create_react_agent(model=openai_model,
                                       prompt=system_prompt, tools=[])

    query_part = [
        HumanMessage(content=state["query"])
    ]

    input_for_agent = {"messages": state["chat_history"] + query_part}
    print(f"______________input for fallback_node_______________:{input_for_agent}")
    result = booking_agent.invoke(input_for_agent)
    for task in state["list_tasks"]:
        if task["agent"] == "fallback_node":
            task["status"] = "done"
            break

    return Command(
        update={
            "messages": [
                AIMessage(
                    content=result["messages"][-1].content,
                    name="fallback_node"
                )
            ]
        },
        goto="supervisor",
    )


def information_node(state: AgentState) -> Command[Literal['validator']]:
    print("*****************called information node************")

    system_prompt = (
        "You are a faq agent.\n\n"
        "INSTRUCTIONS:\n"
        "- You are a consultant specializing in services and information related to the 30Shine men's haircut system. "
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
        "- Chỉ được phép dùng duy nhất 1 tool sau đó handoff cho validation"
        "Phong cách phản hồi:"
        "Thân thiện và gần gũi, xưng là “Janie” hoặc dùng “em” với giọng nhẹ nhàng."
        "Gọi khách hàng là “anh”."
        "Giữ giọng văn nhẹ nhàng, dễ thương, tránh dùng từ “nhé”."
        "Luôn kết thúc câu bằng từ “ạ”."
    )

    system_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt
            ),
            (
                "placeholder",
                "{messages}"
            ),
        ]
    )

    information_agent = create_react_agent(model=openai_model, tools=[get_info, get_near_salon, list_branches],
                                           prompt=system_prompt, version="v2")
    query_part = [
        HumanMessage(content=state["query"])
    ]
    input_for_agent = {"messages": state["chat_history"] + query_part}
    print(f"______________input for information_node_______________:{input_for_agent}")

    result = information_agent.invoke(input_for_agent)
    for task in state["list_tasks"]:
        if task["agent"] == "information_node":
            task["status"] = "done"
            break

    return Command(
        update={
            "messages": [
                AIMessage(
                    content=result["messages"][-1].content,
                    name="information_node"
                )
            ]
        },
        goto="supervisor",
    )


def supervisor_node(state: AgentState) -> Command[Literal['information_node', 'booking_node', 'fallback_node']]:
    print("**************************below is my state right after entering****************************")
    goto = None
    query = None
    if state.get("list_tasks"):
        for task in state["list_tasks"]:
            if task.get("status") != "done":
                goto = task["agent"]
                query = task["task"]
                break

        all_done = all(item.get("status") == "done" for item in state["list_tasks"])
        if all_done:
            goto=END

        return Command(
            goto=goto,
            update={"query": query}
        )


    else:
        user_query = [
            HumanMessage(content=state["query"])
        ]

        messages = [{"role": "system", "content": SUPERVISOR_SYSTEM_PROMPTV4}] + state["chat_history"] + [
            state["messages"][-1]] + user_query

        print(f"______________chat history___________________: \n{state['chat_history']}")

        response = openai_model.with_structured_output(AgentResponse).invoke(messages).response
        print("response", response)


        goto = response[0].agent

        print(f"--- Workflow Transition: Supervisor → {goto.upper()} ---")

        return Command(
            goto=goto,
            update={"list_tasks": [item.dict() for item in response], "query": response[0].task}
        )


class Validator(BaseModel):
    next: Literal["supervisor", "FINISH"] = Field(
        description="Specifies the next worker in the pipeline: 'supervisor' to continue or 'FINISH' to terminate."
    )
    reason: str = Field(
        description="The reason for the decision."
    )
    answer: str = Field(
        description="Base on the user message, agent message answer the user question"
    )


def validator_node(state: AgentState) -> Command[Literal["supervisor", "__end__"]]:
    user_query = [
        HumanMessage(content=state["query"])
    ]

    messages = [
                   {"role": "system", "content": valid_system_prompt}
               ] + state["chat_history"] + state["messages"][1:] + user_query

    response = openai_model.with_structured_output(Validator).invoke(messages)

    goto = response.next
    reason = response.reason

    print(f"REASONING:{reason}\n MESSAGE: {messages}")

    if goto == "FINISH" or goto == END:
        goto = END
        print(" --- Transitioning to END ---")
        print(response.answer)
        return Command(goto=goto, update={'next': goto,
                                          'messages': [AIMessage(content=response.answer)]})
    else:
        print(f"--- Workflow Transition: Validator → Supervisor ---")

        return Command(
            update={
                "messages": [
                    AIMessage(content=reason, name="validator")
                ]
            },
            goto=goto,
        )


graph = StateGraph(AgentState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("information_node", information_node)
graph.add_node("fallback_node", fallback_node)
graph.add_node("booking_node", booking_node)
graph.add_node("validator", validator_node)
graph.add_edge(START, "supervisor")
app = graph.compile()

# import pprint
#
# inputs = {
#     "messages": [
#         ("user", "cho anh hủy lịch cũ vad đặt lịch mới vào sáng mai 9h nhé"),
#     ]
# }
# for event in app.stream(inputs):
#     for key, value in event.items():
#         if value is None:
#             continue
#         pprint.pprint(f"Output from node '{key}':")
#         pprint.pprint(value, indent=2, width=80, depth=None)
#         print()
