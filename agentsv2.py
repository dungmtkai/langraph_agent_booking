from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

from config import BOOKING_SYSTEM_PROMPT, SUPERVISOR_SYSTEM_PROMPTV2
from tools import book_appointment, cancel_appointment, check_availability, get_near_salon, list_branches, faq_answer
from dotenv import load_dotenv
from langgraph.graph import END
from typing import Annotated, TypedDict, Literal, Any
from langgraph.graph import StateGraph, START
from langgraph.types import Command
from datetime import datetime

load_dotenv()
from langgraph.graph.message import add_messages


class Router(TypedDict):
    next: Literal["information_node", "booking_node", "FINISH"]
    reasoning: str
    your_answer: str


class AgentState(TypedDict):
    messages: Annotated[list[Any], add_messages]
    next: str
    query: str
    current_reasoning: str


current_date = datetime.now()

openai_model = ChatOpenAI(model="gpt-4o-mini")

members_dict = {
    'booking_node': 'Hỗ trợ khách hàng trong việc đặt lịch hoặc thay đổi lịch hẹn (không bao gồm email hoặc tên), kiểm tra các khung giờ còn trống tại salon, tìm salon gần nhất và hiển thị các chi nhánh salon. Giao các nhiệm vụ liên quan đến đặt lịch cho trợ lý này.',
    'information_node': 'Cung cấp thông tin tư vấn chi tiết cho khách hàng về các dịch vụ của 30Shine, bảng giá, nhân viên, so sánh giữa các salon, gói combo, tiện ích và chỗ đỗ xe tại cả salon thường và salon cao cấp. Giao các nhiệm vụ liên quan đến câu hỏi thường gặp (FAQ) cho trợ lý này.'}

worker_info = '\n\n'.join([f'WORKER: {member} \nDESCRIPTION: {description}' for member, description in
                           members_dict.items()]) + '\n\nWORKER: FINISH \nDESCRIPTION: If User Query is answered and route to Finished'


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
                                       prompt=system_prompt)

    result = booking_agent.invoke(state)

    return Command(
        update={
            "messages": state["messages"] + [
                AIMessage(content=result["messages"][-1].content, name="booking_node")
                # HumanMessage(content=result["messages"][-1].content, name="booking_node")
            ]
        },
        goto="supervisor",
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

    result = information_agent.invoke(state)

    return Command(
        update={
            "messages": state["messages"] + [
                AIMessage(content=result["messages"][-1].content, name="information_node")
            ]
        },
        goto="supervisor",
    )


def supervisor_node(state: AgentState) -> Command[Literal['information_node', 'booking_node', '__end__']]:
    print("**************************below is my state right after entering****************************")
    print(state)

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

    goto = response.get("next")

    # print("********************************this is my goto*************************")
    # print(goto)
    #
    print("********************************")
    print(query)

    if goto == "FINISH":
        goto = END
        return Command(goto=goto, update={'next': goto,
                                          'current_reasoning': response["reasoning"],
                                          'messages': [AIMessage(content=response["your_answer"])]})

    # print("**************************below is my state****************************")
    # print(state)
    print("**************************below is my answer****************************")
    print(response["your_answer"])

    return Command(goto=goto, update={'next': goto,
                                      'current_reasoning': response["reasoning"]})


graph = StateGraph(AgentState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("information_node", information_node)
graph.add_node("booking_node", booking_node)
graph.add_edge(START, "supervisor")
app = graph.compile(checkpointer=memory)
