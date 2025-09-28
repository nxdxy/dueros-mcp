"""
Classifier Agent

This agent is responsible for:
1. Classify the user's request into a task.
2. Revise the user's request if needed.

The Workflow uses structured output to generate the task and revised request.
"""

from langchain_openai import ChatOpenAI
from assistant_agent.states.state_classifier import TaskClassificationResult
from assistant_agent.prompts import task_classification_prompt
from assistant_agent.states.state_classifier import AgentInputState, AgentState
from assistant_agent.utils import get_today_str, load_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langchain_core.messages import HumanMessage, AIMessage, get_buffer_string


# ===== CONFIGURATION =====

# Initialize model
llm = load_chat_model("openrouter/openai/gpt-4.1-mini")

async def classify_task(state: AgentInputState) -> AgentState:
    prompt = task_classification_prompt.format(messages=state["messages"])
    structured_output_model = llm.with_structured_output(TaskClassificationResult)
    response = await structured_output_model.ainvoke(prompt)
    return {
        "revised_request": response.revised_request,
        "task": response.task,
        "raw_request": response.raw_request,
        "messages": [AIMessage(content=response.revised_request)] if response.task == "no_task" else []
    }
    
# ===== WORKFLOW =====

classifier_graph = StateGraph(AgentInputState)

classifier_graph.add_node("classifier", classify_task)

classifier_graph.add_edge(START, "classifier")
classifier_graph.add_edge("classifier", END)
        
