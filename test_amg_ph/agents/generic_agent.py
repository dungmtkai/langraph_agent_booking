from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from typing import Optional, Dict, Any, List
from .settings import AgentSetting, get_agent_setting
from ..tools.tool_agent import create_langchain_tool


class GenericAgent:
    def __init__(
        self,
        setting: AgentSetting,
        user_info: Optional[Dict[str, Any]] = None,
    ):
        self.setting = setting
        self.user_info = user_info or {}

        self.tools: List[StructuredTool] = [
            create_langchain_tool(tool_id) for tool_id in setting.tool_ids
        ]

        self.llm = ChatOpenAI(
            model=setting.model_config.name,
            temperature=setting.model_config.temperature
        )

    @classmethod
    def from_id(cls, agent_id: int, **kwargs) -> "GenericAgent":
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
