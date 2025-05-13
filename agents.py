from langgraph.prebuilt import create_react_agent
from tools import book_appointment, cancel_appointment, check_availability, get_near_salon, list_branches, faq_answer


booking_agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[book_appointment, cancel_appointment, check_availability, get_near_salon, list_branches],
    prompt=(
        "You are a booking agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with research-related tasks, DO NOT do any math\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="research_agent",
)


faq_agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[faq_answer],
    prompt=(
        "You are a research agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with research-related tasks, DO NOT do any math\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="research_agent",
)