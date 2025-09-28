"""
Full Multi-Agent Assistant System

This system is responsible for:
1. Classifying the user's request into a task.
2. Revising the user's request if needed.
3. Executing the task.

The Workflow uses structured output to generate the task and revised request.
"""

from langgraph.graph import StateGraph, START, END
from assistant_agent.classifier import classify_task
from assistant_agent.xiaodu_agent import xiaodu_assistant_agent
from assistant_agent.daily_chat_agent import daily_chatter_agent
from assistant_agent.common_task_agent import common_task_agent
from assistant_agent.states.state_classifier import AgentInputState, AgentState
from typing_extensions import Literal

def decide_next_node(state:AgentState) -> Literal["xiaodu_assistant", "daily_chatter", "common_task_agent", "no_task"]:
    """This node will select the next node of the graph"""
    return state["task"]

# ===== GRAPH CONSTRUCTION =====
graph = StateGraph(AgentState, input_schema=AgentInputState)
graph.add_node("classifier", classify_task)
graph.add_node("xiaodu_assistant", xiaodu_assistant_agent)
graph.add_node("daily_chatter", daily_chatter_agent)
graph.add_node("common_task_agent", common_task_agent)
graph.add_edge(START, "classifier")
graph.add_conditional_edges(
    "classifier",
    decide_next_node,
    {
        "xiaodu_assistant": "xiaodu_assistant",
        "daily_chatter": "daily_chatter",
        "common_task_agent": "common_task_agent",
        "no_task": END
    }
)
graph.add_edge("xiaodu_assistant", END)
graph.add_edge("daily_chatter", END)

agent = graph.compile()

