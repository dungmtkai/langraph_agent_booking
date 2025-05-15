from langchain.agents import AgentExecutor
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

from config import BOOKING_SYSTEM_PROMPT, SUPERVISOR_SYSTEM_PROMPTV2, SUPERVISOR_SYSTEM_PROMPTV3, valid_system_prompt
from tools import book_appointment, cancel_appointment, check_availability, get_near_salon, list_branches, faq_answer
from dotenv import load_dotenv
from langgraph.graph import END
from typing import Annotated, TypedDict, Literal, Any, Optional
from langgraph.graph import StateGraph, START
from langgraph.types import Command
from datetime import datetime

load_dotenv()
from langgraph.graph.message import add_messages, MessagesState

from pydantic import BaseModel, Field

next_des = ("Determines which specialist to activate next in the workflow sequence:"
            "'booking_node' Hỗ trợ khách hàng trong việc đặt lịch hoặc thay đổi lịch hẹn (không bao gồm email hoặc tên), kiểm tra các khung giờ còn trống tại salon, tìm salon gần nhất và hiển thị các chi nhánh salon. Giao các nhiệm vụ liên quan đến đặt lịch cho trợ lý này."
            "'information_node' Cung cấp thông tin tư vấn chi tiết cho khách hàng về các dịch vụ của 30Shine, bảng giá, nhân viên, so sánh giữa các salon, gói combo, tiện ích và chỗ đỗ xe tại cả salon thường và salon cao cấp. Giao các nhiệm vụ liên quan đến câu hỏi thường gặp (FAQ) cho trợ lý này."
            )


class AgentState(TypedDict):
    messages: Annotated[list[Any], add_messages]
    query: str
    chat_history: Annotated[list[Any], add_messages]


class Router(BaseModel):
    next: Literal["information_node", "booking_node"] = Field(description=next_des)
    reason: str = Field(
        description="Detailed justification for the routing decision, explaining the rationale behind selecting the particular specialist and how this advances the task toward completion."
    )


current_date = datetime.now()

openai_model = ChatOpenAI(model="gpt-4o-mini")

members_dict = {
    'booking_node': 'Hỗ trợ khách hàng trong việc đặt lịch hoặc thay đổi lịch hẹn (không bao gồm email hoặc tên), kiểm tra các khung giờ còn trống tại salon, tìm salon gần nhất và hiển thị các chi nhánh salon. Giao các nhiệm vụ liên quan đến đặt lịch cho trợ lý này.',
    'information_node': 'Cung cấp thông tin tư vấn chi tiết cho khách hàng về các dịch vụ của 30Shine, bảng giá, nhân viên, so sánh giữa các salon, gói combo, tiện ích và chỗ đỗ xe tại cả salon thường và salon cao cấp. Giao các nhiệm vụ liên quan đến câu hỏi thường gặp (FAQ) cho trợ lý này.'}

worker_info = '\n\n'.join([f'WORKER: {member} \nDESCRIPTION: {description}' for member, description in
                           members_dict.items()])


def booking_node(state: AgentState) -> Command[Literal['supervisor']]:
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
                                       tools=[book_appointment, cancel_appointment, check_availability, get_near_salon,
                                              list_branches],
                                       debug=True,
                                       prompt=system_prompt)

    input_for_agent = {"messages": state["chat_history"] + [state["query"]]}
    print(f"______________input for booking_node_______________:{input_for_agent}")
    result = booking_agent.invoke(input_for_agent)

    return Command(
        update={
            "messages": [
                AIMessage(
                    content=result["messages"][-1].content,
                    name="booking_node"
                )
            ]
        },
        goto="validator",
    )


def information_node(state: AgentState) -> Command[Literal['supervisor']]:
    print("*****************called information node************")

    system_prompt = (
        "You are a faq agent.\n\n"
        "INSTRUCTIONS:\n"
        "- You are a consultant specializing in services and information related to the 30Shine men's haircut system. "
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
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

    information_agent = create_react_agent(model=openai_model, tools=[faq_answer, get_near_salon, list_branches],
                                           prompt=system_prompt)

    input_for_agent = {"messages": state["chat_history"] + [state["query"]]}
    print(f"______________input for information_node_______________:{input_for_agent}")

    result = information_agent.invoke(input_for_agent)

    return Command(
        update={
            "messages": [
                AIMessage(
                    content=result["messages"][-1].content,
                    name="information_node"
                )
            ]
        },
        goto="validator",
    )


def supervisor_node(state: AgentState) -> Command[Literal['information_node', 'booking_node']]:
    print("**************************below is my state right after entering****************************")
    user_query = [
        HumanMessage(content=state["query"], name="validator")
    ]

    messages = [{"role": "system", "content": SUPERVISOR_SYSTEM_PROMPTV3}] + state["chat_history"] + user_query

    print(f"______________chat history___________________: \n{state['chat_history']}")

    response = openai_model.with_structured_output(Router).invoke(messages)

    goto = response.next

    print(f"--- Workflow Transition: Supervisor → {goto.upper()} ---")

    return Command(
        update={
            "query": state["query"]
        },
        goto=goto,
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
    messages = [
                   {"role": "system", "content": valid_system_prompt}
               ] + state["chat_history"] + state["messages"][1:] + [state["query"]]

    response = openai_model.with_structured_output(Validator).invoke(messages)

    goto = response.next
    reason = response.reason

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
