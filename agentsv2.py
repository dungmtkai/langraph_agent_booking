from langchain.agents import AgentExecutor
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

from config import BOOKING_SYSTEM_PROMPT, SUPERVISOR_SYSTEM_PROMPTV2
from tools import book_appointment, cancel_appointment, check_availability, get_near_salon, list_branches, get_info
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


class Router(BaseModel):
    next: Literal["information_node", "booking_node", "FINISH"] = Field(description=next_des)
    reason: str = Field(
        description="Detailed justification for the routing decision, explaining the rationale behind selecting the particular specialist and how this advances the task toward completion."
    )
    answer: str = Field(
        description="Base on the user message, agent message answer the user question"
    )


current_date = datetime.now()

openai_model = ChatOpenAI(model="gpt-4o-mini")

members_dict = {
    'booking_node': 'Hỗ trợ khách hàng trong việc đặt lịch hoặc thay đổi lịch hẹn (không bao gồm email hoặc tên), kiểm tra các khung giờ còn trống tại salon, tìm salon gần nhất và hiển thị các chi nhánh salon. Giao các nhiệm vụ liên quan đến đặt lịch cho trợ lý này.',
    'information_node': 'Cung cấp thông tin tư vấn chi tiết cho khách hàng về các dịch vụ của 30Shine, bảng giá, nhân viên, so sánh giữa các salon, gói combo, tiện ích và chỗ đỗ xe tại cả salon thường và salon cao cấp. Giao các nhiệm vụ liên quan đến câu hỏi thường gặp (FAQ) cho trợ lý này.'}

worker_info = '\n\n'.join([f'WORKER: {member} \nDESCRIPTION: {description}' for member, description in
                           members_dict.items()]) + '\n\nWORKER: FINISH \nDESCRIPTION: If User Query is answered and route to Finished'


def booking_node(state: MessagesState) -> Command[Literal['supervisor']]:
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
                                       prompt=system_prompt)
    result = booking_agent.invoke(state["messages"][-1])

    return Command(
        update={
            "messages": state["messages"] + [
                AIMessage(content=result["messages"][-1].content, name="booking_node")
            ]
        },
        goto="supervisor",
    )


def information_node(state: MessagesState) -> Command[Literal['supervisor']]:
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

    information_agent = create_react_agent(model=openai_model, tools=[get_info, get_near_salon, list_branches],
                                           prompt=system_prompt)

    result = information_agent.invoke(state["messages"][-1])

    return Command(
        update={
            "messages": state["messages"] + [
                AIMessage(content=result["messages"][-1].content, name="information_node")
            ]
        },
        goto="supervisor",
    )


def supervisor_node(state: MessagesState) -> Command[Literal['information_node', 'booking_node', '__end__']]:
    print("**************************below is my state right after entering****************************")

    messages = [
                   {"role": "system", "content": SUPERVISOR_SYSTEM_PROMPTV2.format(worker_info=worker_info)}
               ] + state["messages"]

    # print("***********************this is my message*****************************************")
    # print(messages)

    query = state['messages'][-1].content if state["messages"] else ""
    query = ''
    if len(state['messages']) == 1:
        query = state['messages'][0].content

    response = openai_model.with_structured_output(Router).invoke(messages)

    print(response)

    goto = response.next

    # print("********************************this is my goto*************************")
    # print(goto)
    #
    print("********************************")
    print(query)

    if goto == "FINISH":
        goto = END
        print("********************************this is my anwser*************************")
        print(response.answer)
        return Command(goto=goto, update={'next': goto,
                                          'messages': [AIMessage(content=response.answer)]})

    # print("**************************below is my state****************************")
    # print(state)
    print("**************************below is my answer****************************")
    # print(response["your_answer"])

    return Command(
        update={
            "messages": [
                HumanMessage(content=response.reason, name="supervisor")
            ]
        },
        goto=goto,
    )


graph = StateGraph(MessagesState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("information_node", information_node)
graph.add_node("booking_node", booking_node)
graph.add_edge(START, "supervisor")
app = graph.compile()

import pprint

inputs = {
    "messages": [
        ("user", "Chào bạn, mình muốn đặt lịch cắt tóc nam vào cuối tuần này, không biết salon còn lịch không nhỉ?"),
    ]
}
for event in app.stream(inputs):
    for key, value in event.items():
        if value is None:
            continue
        pprint.pprint(f"Output from node '{key}':")
        pprint.pprint(value, indent=2, width=80, depth=None)
        print()
