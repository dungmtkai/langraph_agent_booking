from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.types import Send
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langgraph_supervisor import create_supervisor

from config import BOOKING_SYSTEM_PROMPT
from tools import book_appointment, cancel_appointment, check_availability, get_near_salon, list_branches, faq_answer
from dotenv import load_dotenv
from langgraph.graph import END
from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.types import Command
from utils import pretty_print_messages
from datetime import datetime

load_dotenv()
current_date = datetime.now()
booking_agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[book_appointment, cancel_appointment, check_availability, get_near_salon, list_branches],
    prompt=BOOKING_SYSTEM_PROMPT.format(date_time=current_date),
    name="booking_agent",
)

faq_agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[faq_answer],
    prompt=(
        "You are a faq agent.\n\n"
        "INSTRUCTIONS:\n"
        "- You are a consultant specializing in services and information related to the 30Shine men's haircut system. "
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="faq_agent",
)


def create_handoff_tool(*, agent_name: str, description: str | None = None):
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help."

    @tool(name, description=description)
    def handoff_tool(
        state: Annotated[MessagesState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",
            "name": name,
            "tool_call_id": tool_call_id,
        }
        # highlight-next-line
        return Command(
            # highlight-next-line
            goto=agent_name,  # (1)!
            # highlight-next-line
            update={**state, "messages": state["messages"] + [tool_message]},  # (2)!
            # highlight-next-line
            graph=Command.PARENT,  # (3)!
        )

    return handoff_tool


# Handoffs
assign_to_booking_agent = create_handoff_tool(
    agent_name="booking_agent",
    description="Assign task to a booking agent.",
)

assign_to_faq_agent = create_handoff_tool(
    agent_name="faq_agent",
    description="Assign task to a faq agent.",
)

# supervisor_agent_with_description = create_react_agent(
#     model="openai:gpt-4o-mini",
#     tools=[
#         assign_to_booking_agent,
#         assign_to_faq_agent,
#     ],
#     prompt=(
#         "Bạn là Janie, trợ lý ảo tư vấn và chăm sóc khách hàng của 30Shine, một chuỗi salon và hairdressing dành cho nam giới và trẻ nhỏ."
#         "You are managing two agents:\n"
#         "- a booking agent. Support customers in booking or rescheduling appointments (not email or name), and checking available time slots at salons. Assign booking tasks to this assistant\n"
#         "- a faq agent. Provides customers with consultation and detailed information about 30Shine's services, pricing, staff, salon comparisons, combos, amenities, and parking across regular and premium salons. Assign faq tasks to this assistant\n"
#         "Chỉ giao task cho một agent duy nhất tại một thời điểm"
#         "Sau khi một agent hoàn tất công việc, bạn cần đọc message cuối cùng trong đoạn hội thoại và tổng hợp hoặc phản hồi lại cho người dùng.\n.\n"
#     ),
#     name="supervisor",
# )
supervisor = create_supervisor(
    agents=[faq_agent, booking_agent],
    model=ChatOpenAI(model="gpt-4o-mini"),
    prompt=(
                "Bạn là Janie, trợ lý ảo tư vấn và chăm sóc khách hàng của 30Shine, một chuỗi salon và hairdressing dành cho nam giới và trẻ nhỏ."
                "You are managing two agents:\n"
                "- a booking agent. Support customers in booking or rescheduling appointments (not email or name), and checking available time slots at salons. Assign booking tasks to this assistant\n"
                "- a faq agent. Provides customers with consultation and detailed information about 30Shine's services, pricing, staff, salon comparisons, combos, amenities, and parking across regular and premium salons. Assign faq tasks to this assistant\n"
                "Chỉ giao task cho một agent duy nhất tại một thời điểm"
                "Sau khi một agent hoàn tất công việc, bạn cần đọc message cuối cùng trong đoạn hội thoại và tổng hợp hoặc phản hồi lại cho người dùng.\n.\n"
            ),
    parallel_tool_calls=True
).compile()
# supervisor_with_description = (
#     StateGraph(MessagesState)
#     .add_node(
#         supervisor_agent_with_description, destinations=("booking_agent", "faq_agent", END)
#     )
#     .add_node(booking_agent)
#     .add_node(faq_agent)
#     .add_edge(START, "supervisor")
#     .add_edge("booking_agent", "supervisor")
#     .add_edge("faq_agent", "supervisor")
#     .compile()
# )

for chunk in supervisor.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Tôi muốn đặt lịch cắt tóc lúc 10h sáng mai và giá ở 30 shine như nào",
                }
            ]
        },
        subgraphs=True,
):
    pretty_print_messages(chunk, last_message=True)


# final_message_history = chunk["supervisor"]["messages"]
