from typing import Dict, List, Optional, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver

from . import SupervisorState, Action, SupervisorPlan
from .generic_agent import GenericAgent
from .generic_agent_settings import ModelConfig
from .prompts import AGENT_NODE_TO_ID, build_supervisor_system_prompt


def _get_agent_registry(user_info: Optional[Dict[str, Any]] = None) -> Dict[str, GenericAgent]:
    """Create agent registry with optional user_info"""
    return {
        node_name: GenericAgent.from_id(agent_id, user_info=user_info)
        for node_name, agent_id in AGENT_NODE_TO_ID.items()
    }


def group_tasks_by_agent(actions: List[Action], original_query: str) -> List[dict]:
    """Group multiple tasks for same agent into 1 task with original query"""
    seen_agents = set()
    grouped_tasks = []

    for action in actions:
        if action.name not in seen_agents:
            seen_agents.add(action.name)
            grouped_tasks.append({
                "name": action.name,
                "query": original_query,
                "status": "pending"
            })

    return grouped_tasks


def create_supervisor_node(llm: ChatOpenAI):
    """Factory to create supervisor node with specified LLM"""

    def supervisor_node(state: SupervisorState):
        print("\n" + "#" * 60)
        print("[SUPERVISOR] Supervisor đang xử lý...")
        print(f"[SUPERVISOR] Current tasks: {state.get('list_tasks', [])}")

        # Check pending tasks - continue with first unfinished task
        for task in state.get("list_tasks", []):
            if task.get("status") != "done":
                print(f"[SUPERVISOR] → Tiếp tục task chưa hoàn thành: {task['name']}")
                print("#" * 60)
                return Command(goto=task["name"], update={"query": task["query"]})

        # All tasks done → END
        if state.get("list_tasks") and all(t.get("status") == "done" for t in state["list_tasks"]):
            print("[SUPERVISOR] ✅ Tất cả tasks hoàn thành → END")
            print("#" * 60)
            return Command(goto=END)

        # Analyze new query
        print("[SUPERVISOR] Phân tích query mới...")
        system_prompt = build_supervisor_system_prompt()
        messages = [{"role": "system", "content": system_prompt}] + state["messages"]
        response = llm.with_structured_output(SupervisorPlan).invoke(messages)
        print(f"[SUPERVISOR] Response: {response}")

        original_query = state.get("original_query") or state.get("query", "")
        tasks = group_tasks_by_agent(response.actions, original_query)

        print(f"[SUPERVISOR] Raw actions: {[a.name for a in response.actions]}")
        print(f"[SUPERVISOR] Grouped tasks: {[t['name'] for t in tasks]}")

        if not tasks:
            print("[SUPERVISOR] Không có tasks → END")
            print("#" * 60)
            return Command(goto=END)

        first_task = tasks[0]
        print(f"[SUPERVISOR] → Chuyển đến: {first_task['name']}")
        print("#" * 60)
        return Command(
            goto=first_task["name"],
            update={"list_tasks": tasks, "query": first_task["query"]}
        )

    return supervisor_node


def create_agent_node(node_name: str, agent_registry: Dict[str, GenericAgent]):
    """Factory function to create agent node"""

    def agent_node(state: SupervisorState):
        print("\n" + "=" * 60)
        print(f"[{node_name.upper()}] Bắt đầu xử lý...")
        print(f"[{node_name.upper()}] Query: {state['query']}")

        agent = agent_registry[node_name]
        chat_history = state.get("chat_history", [])

        result = agent.invoke(
            messages=chat_history + [HumanMessage(content=state["query"])]
        )

        final_msg = result["messages"][-1].content
        print(f"[{node_name.upper()}] Kết quả: {final_msg[:100]}...")

        # Mark task as done
        for t in state["list_tasks"]:
            if t["name"] == node_name:
                t["status"] = "done"
                print(f"[{node_name.upper()}] ✅ Đánh dấu task done")

        print(f"[{node_name.upper()}] → Quay về supervisor")
        print("=" * 60)

        return Command(
            update={"messages": [AIMessage(content=final_msg, name=node_name)]},
            goto="supervisor"
        )

    return agent_node


def build_workflow(
    user_info: Optional[Dict[str, Any]] = None,
    supervisor_model: Optional[ModelConfig] = None
):
    """Build supervisor workflow with all agent nodes

    Args:
        user_info: Optional user info dict to pass to all agents
        supervisor_model: Optional ModelConfig for supervisor LLM (default: gpt-4.1-mini)

    Returns:
        Compiled LangGraph workflow
    """
    agent_registry = _get_agent_registry(user_info)

    # Default to gpt-5-mini
    model_config = supervisor_model or ModelConfig(name="gpt-5-mini", temperature=0)
    extra_kwargs = model_config.get_extra_kwargs()

    llm = ChatOpenAI(
        model=model_config.name,
        temperature=model_config.temperature,
        **extra_kwargs
    )

    workflow = StateGraph(SupervisorState)

    workflow.add_node("supervisor", create_supervisor_node(llm))

    for node_name in agent_registry.keys():
        workflow.add_node(node_name, create_agent_node(node_name, agent_registry))

    workflow.add_edge(START, "supervisor")

    return workflow.compile(checkpointer=MemorySaver())
