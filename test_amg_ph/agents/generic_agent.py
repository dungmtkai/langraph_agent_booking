from typing import Optional, Dict, Any, List

from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

from .generic_agent_settings import AgentSetting, get_agent_setting
from .state import GenericAgentState
from .schemas import GenericAgentPlan
from .prompts import (
    get_tool_descriptions,
    render_description,
    create_planner_system_prompt,
    create_tool_executor_system_prompt,
    RESPONDER_SYSTEM_PROMPT,
)
from ..tools.tool_agent import create_langchain_tool


DEFAULT_USER_INFO = {
    "parentName": "Mai Thị Kim Dung",
    "relationship": "MOTHER",
    "childName": "Nguyễn Thị Mai",
    "nickname": "Chíp",
    "className": "Chồi BU1",
    "teacherName": "Nguyễn Thị Hà",
}


class GenericAgent:
    def __init__(
        self,
        setting: AgentSetting,
        user_info: Optional[Dict[str, Any]] = None,
    ):
        self.setting = setting
        self.user_info = {**DEFAULT_USER_INFO, **(user_info or {})}

        self.tools: List[StructuredTool] = [
            create_langchain_tool(tool_id) for tool_id in setting.tool_ids
        ]
        self.tool_map = {t.name: t for t in self.tools}

        self.llm = ChatOpenAI(
            model=setting.model_config.name,
            temperature=setting.model_config.temperature
        )

        self._graph = None

    @classmethod
    def from_id(cls, agent_id: int, **kwargs) -> "GenericAgent":
        setting = get_agent_setting(agent_id)
        if not setting:
            raise ValueError(f"Agent with id {agent_id} not found")
        return cls(setting=setting, **kwargs)

    def planner(self, state: GenericAgentState) -> dict:
        messages = state["messages"]
        system_prompt = create_planner_system_prompt(
            description=render_description(self.setting.description, self.user_info),
            tool_descriptions=get_tool_descriptions(self.tools),
            instruction=self.setting.instruction,
            rule=self.setting.rule
        )

        print(f"[Plan system prompt]: {system_prompt}")

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("placeholder", "{messages}")
        ])

        chain = prompt | self.llm.with_structured_output(GenericAgentPlan, method="function_calling")

        try:
            plan: GenericAgentPlan = chain.invoke({"messages": messages})
        except Exception as e:
            print(f"[PLANNER][{self.setting.name}] Error: {e}")
            plan = GenericAgentPlan(
                thought="Lỗi parse, cần hỏi lại",
                next_action="ask_user",
                response="Dạ anh ơi, em chưa hiểu lắm, anh nói rõ hơn được không ạ?"
            )

        pending_tools = [t.tool_name for t in plan.selected_tools] if plan.next_action == "call_tool" else []

        print(f"\n[PLANNER][{self.setting.name}]")
        print(f"  Thought: {plan.thought}")
        print(f"  Next action: {plan.next_action}")
        if pending_tools:
            print(f"  Selected tools: {pending_tools}")
        if plan.response:
            print(f"  Response: {plan.response[:100]}...")

        return {
            "plan": plan.model_dump(),
            "pending_tools": pending_tools
        }

    def tool_executor(self, state: GenericAgentState) -> dict:
        pending_tools = state.get("pending_tools", [])
        messages = state["messages"]

        if not pending_tools:
            return {"pending_tools": []}

        selected_tools = [self.tool_map[name] for name in pending_tools if name in self.tool_map]

        if not selected_tools:
            return {"pending_tools": []}

        system_prompt = create_tool_executor_system_prompt(pending_tools)

        llm_with_tools = ChatOpenAI(
            model=self.setting.model_config.name,
            temperature=0
        ).bind_tools(selected_tools, tool_choice="required")

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("placeholder", "{messages}")
        ])

        try:
            response = (prompt | llm_with_tools).invoke({"messages": messages})
            new_messages = []

            if hasattr(response, 'tool_calls') and response.tool_calls:
                new_messages.append(response)

                for tc in response.tool_calls:
                    tool_name = tc['name']
                    tool_args = tc['args']
                    tool_call_id = tc['id']

                    print(f"\n[TOOL_EXECUTOR][{self.setting.name}]")
                    print(f"  Calling: {tool_name}")
                    print(f"  Args: {tool_args}")

                    if tool_name in self.tool_map:
                        try:
                            result = self.tool_map[tool_name].invoke(tool_args)
                            print(f"  Result: {str(result)[:200]}...")
                        except Exception as e:
                            result = f"Lỗi khi gọi {tool_name}: {e}"
                            print(f"  Error: {e}")
                    else:
                        result = f"Tool {tool_name} không tồn tại"
                        print(f"  Error: Tool not found")

                    new_messages.append(ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call_id,
                        name=tool_name
                    ))

        except Exception as e:
            print(f"[TOOL_EXECUTOR][{self.setting.name}] Error: {e}")
            new_messages = []

        return {
            "messages": new_messages,
            "pending_tools": []
        }

    def responder(self, state: GenericAgentState) -> dict:
        plan = state.get("plan", {})
        response = plan.get("response", "")

        if response:
            text = response
        else:
            prompt = ChatPromptTemplate.from_messages([
                ("system", RESPONDER_SYSTEM_PROMPT),
                ("placeholder", "{messages}")
            ])
            text = (prompt | self.llm).invoke({"messages": state["messages"]}).content

        return {"messages": [AIMessage(content=text, name=self.setting.name)]}

    def router(self, state: GenericAgentState) -> str:
        plan = state.get("plan", {})
        next_action = plan.get("next_action", "respond")

        if next_action == "call_tool":
            return "tool_executor"
        return "respond"

    def build_graph(self):
        if self._graph is not None:
            return self._graph

        g = StateGraph(GenericAgentState)

        g.add_node("planner", self.planner)
        g.add_node("tool_executor", self.tool_executor)
        g.add_node("respond", self.responder)

        g.add_edge(START, "planner")
        g.add_conditional_edges(
            "planner",
            self.router,
            {"tool_executor": "tool_executor", "respond": "respond"}
        )
        g.add_edge("tool_executor", "planner")
        g.add_edge("respond", END)

        self._graph = g.compile()
        return self._graph

    def invoke(self, messages: list, **kwargs) -> dict:
        graph = self.build_graph()
        return graph.invoke({
            "messages": messages,
            "plan": {},
            "pending_tools": []
        }, **kwargs)
