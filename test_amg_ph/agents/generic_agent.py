from langchain_openai import ChatOpenAI
from typing import Optional, Dict, Any, List
from .settings import AgentSetting, get_agent_setting


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class GenericAgent:
    def __init__(
        self,
        setting: AgentSetting,
        user_info: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Any]] = None
    ):
        self.setting = setting
        self.user_info = user_info or {}
        self.tools = tools or []

    @classmethod
    def from_id(cls, agent_id: int, **kwargs) -> "GenericAgent":
        """Create agent from ID"""
        setting = get_agent_setting(agent_id)
        if not setting:
            raise ValueError(f"Agent with id {agent_id} not found")
        return cls(setting=setting, **kwargs)


def create_absence_request_agent(**kwargs) -> GenericAgent:
    return GenericAgent.from_id(159, **kwargs)


def create_daily_report_agent(**kwargs) -> GenericAgent:
    return GenericAgent.from_id(160, **kwargs)


def create_feedback_agent(**kwargs) -> GenericAgent:
    return GenericAgent.from_id(161, **kwargs)


def create_get_submitted_ticket_agent(**kwargs) -> GenericAgent:
    return GenericAgent.from_id(162, **kwargs)


def create_learning_schedule_agent(**kwargs) -> GenericAgent:
    return GenericAgent.from_id(163, **kwargs)


def create_meal_info_agent(**kwargs) -> GenericAgent:
    return GenericAgent.from_id(164, **kwargs)


def create_medication_instruction_agent(**kwargs) -> GenericAgent:
    return GenericAgent.from_id(165, **kwargs)


def create_pickup_authorization_agent(**kwargs) -> GenericAgent:
    return GenericAgent.from_id(166, **kwargs)
