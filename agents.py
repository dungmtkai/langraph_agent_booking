from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
from langgraph_supervisor import create_supervisor

from config import BOOKING_SYSTEM_PROMPT, SUPERVISOR_SYSTEM_PROMPT
from tools import book_appointment, cancel_appointment, check_availability, get_near_salon, list_branches, get_info
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
    version="v2"
)

infor_agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[get_info, get_near_salon, list_branches],
    prompt=(
        "You are a faq agent.\n\n"
        "INSTRUCTIONS:\n"
        "- You are a consultant specializing in services and information related to the 30Shine men's haircut system. "
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="infor_agent",
    version="v2"
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

        print("state")
        # highlight-next-line
        return Command(
            # highlight-next-line
            goto=agent_name,  # (1)!
            # highlight-next-line
            update={**state, "messages": state["messages"]},  # (2)!
            # highlight-next-line
            graph=Command.PARENT,  # (3)!
        )

    return handoff_tool


supervisor = create_supervisor(
    agents=[infor_agent, booking_agent],
    model=ChatOpenAI(model="gpt-4o-mini"),
    prompt=SUPERVISOR_SYSTEM_PROMPT,
    add_handoff_messages=False,
    output_mode="full_history"
).compile(checkpointer=memory)
