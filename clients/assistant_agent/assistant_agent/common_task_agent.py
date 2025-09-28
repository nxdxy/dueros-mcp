"""
Common Task Agent

处理适合通过工具调用来满足的日常任务（小度设备相关的任务有独立的Agent处理 和当前Agent无交集）

"""

from assistant_agent.utils import load_chat_model
from assistant_agent.states.state_common_task import CommonTaskState, CommonTaskOutputState, CommonTaskClassifation

from assistant_agent.prompts.common_task_prompts import (
    COMMON_TASK_PROMPT, COMMON_TASK_RESPONSE_PROMPT,
    COMMON_TASK_INITIAL_SUMMARY_PROMPT, COMMON_TASK_EXISTING_SUMMARY_PROMPT, COMMON_TASK_FINAL_SUMMARY_PROMPT,COMMON_TASK_CLASSIFICATION_PROMPT
    )
from assistant_agent.tools.mcp_manager import get_mcp_tools, get_mcp_tools_sync
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from pydantic import BaseModel, Field
from typing_extensions import Optional
from langgraph.types import interrupt
import json
from langmem.short_term import SummarizationNode
from langchain_core.messages.utils import count_tokens_approximately
from assistant_agent.utils import get_today_str

# ===== CONFIGURATION =====
# Initialize models
summary_llm = load_chat_model("openrouter/openai/gpt-4o-mini")
classifier_llm = load_chat_model("openrouter/openai/gpt-4o-mini")  # 用于任务分类
simple_task_llm = load_chat_model("openrouter/openai/gpt-4o")  # 普通任务模型
complex_task_llm = load_chat_model("openrouter/anthropic/claude-sonnet-4")  # 复杂任务模型

# Create SummarizationNode instance
summarization_model = summary_llm.bind(max_tokens=2048)  # 限制摘要模型输出长度
summarization_node = SummarizationNode(
    token_counter=count_tokens_approximately,
    model=summarization_model,
    max_tokens=4096,  # 消息历史最大token数
    max_summary_tokens=2048,  # 摘要最大token数  
    output_messages_key="llm_input_messages",  # 输出到此键，不覆盖原始消息,
    initial_summary_prompt=COMMON_TASK_INITIAL_SUMMARY_PROMPT,
    existing_summary_prompt=COMMON_TASK_EXISTING_SUMMARY_PROMPT,
    final_prompt=COMMON_TASK_FINAL_SUMMARY_PROMPT)

class AskHuman(BaseModel):
    """
    当你有任何不确定的信息需要用户协助时，调用该工具提问，用户将回答你的问题
    """
    question: Optional[str] = Field(
        description="向用户提问的问题"
    )


# ===== AGENT NODES =====
async def task_classification(state: CommonTaskState) -> dict:
    """
    任务分类节点：分析用户请求的复杂度
    
    Args:
        state: 当前状态
    
    Returns:
        dict: 包含task_complexity字段的更新状态
    """
    
    # 使用分类模型判断任务复杂度
    classification_prompt = COMMON_TASK_CLASSIFICATION_PROMPT.format(
        messages=state["messages"]
    )
    
    structured_classifier_model = classifier_llm.with_structured_output(CommonTaskClassifation)

    response = await structured_classifier_model.ainvoke(classification_prompt)
    
    # 从结构化响应中获取复杂度
    complexity = response.task_complexity
    
    # 确保返回值有效
    if complexity not in ["simple", "complex"]:
        complexity = "simple"  # 默认为简单任务
    
    return {"task_complexity": complexity}

async def llm_call(state: CommonTaskState) -> dict:
    """Analyze current state and decide on tool usage with MCP integration.

    This node:
    1. Uses summarized messages if available (from summarization_node)
    2. Selects appropriate model based on task complexity
    3. Retrieves available tools from MCP server
    4. Binds tools to the language model
    5. Processes user input and decides on tool usage

    Returns updated state with model response.
    """
    user_info = state.get("user_info")
    user_name_v = user_info.get("user_name", "用户") if user_info else "用户"
    location_v = user_info.get("location", "未知位置") if user_info else "未知位置"
    
    # Use summarized messages if available, otherwise use original messages
    input_messages = state.get("llm_input_messages", state["messages"])
    
    prompt = COMMON_TASK_PROMPT.format(
        user_name=user_name_v, 
        location=location_v, 
        date=get_today_str(),
        messages=input_messages
    )
    
    # Select model based on task complexity
    task_complexity = state.get("task_complexity", "simple")
    if task_complexity == "complex":
        selected_llm = complex_task_llm
    else:
        selected_llm = simple_task_llm
    
    # Get available tools from MCP server
    mcp_tools = await get_mcp_tools()
    
    # Add ask_human_input tool to available tools
    tools = mcp_tools + [AskHuman]

    #import pdb
    #pdb.set_trace()
    # Initialize model with tool binding
    model_with_tools = selected_llm.bind_tools(tools)
    messages = await model_with_tools.ainvoke(prompt)
    
    return {
        "messages": messages
    }

async def human_node(state: CommonTaskState) -> dict:
    """
    Human-in-the-loop节点，处理需要人工干预的情况
    
    Args:
        state: 当前状态
    
    Returns:
        dict: 更新后的状态
    """
    tool_call_id = state["messages"][-1].tool_calls[0]["id"]
    question = AskHuman.model_validate(state["messages"][-1].tool_calls[0]["args"])

    user_response = interrupt(question)
    tool_message = [ToolMessage(tool_call_id=tool_call_id, content=user_response)]
    return {"messages": tool_message}

async def polish_response(state: CommonTaskState) -> dict:
    """
    从消息中提取用户展示的回复内容，只包含response部分
    
    Args:
        state: 当前状态
    
    Returns:
        dict: 包含user_response字段的更新状态
    """
    user_info = state.get("user_info")
    user_name_v = user_info.user_name if user_info else "用户"
    location_v = user_info.location if user_info else "未知位置"
    
    # Use summarized messages if available, otherwise use original messages
    input_messages = state.get("llm_input_messages", state["messages"])
    
    prompt = COMMON_TASK_RESPONSE_PROMPT.format(
        user_name=user_name_v, 
        location=location_v, 
        messages=input_messages
    )

    messages = await complex_task_llm.ainvoke(prompt)
        
    return {"user_response": messages}

# ===== ROUTING LOGIC =====
def should_continue(state: CommonTaskState):
    """Determine if the workflow should continue.

    This function:
    1. Checks if the last message has tool calls -> analyze and route appropriately
    2. Otherwise process output and end the workflow
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # Check if the last message has tool calls (from LLM)
    if not last_message.tool_calls:    
        return "polish_response"
    elif last_message.tool_calls[0]["name"] == "AskHuman":
        return "ask_human"
    # Otherwise if there is, we continue
    else:
        return "tool_node"

# ===== GRAPH CONSTRUCTION =====
graph = StateGraph(CommonTaskState, output_schema=CommonTaskOutputState)
graph.add_node("task_classification", task_classification)
graph.add_node("summarization", summarization_node)
graph.add_node("llm_call", llm_call)
graph.add_node("human_node", human_node)
graph.add_node("polish_response", polish_response)

# Create standard tool node with MCP tools + ask_human_input tool
mcp_tools = get_mcp_tools_sync()
common_task_tool_node = ToolNode(tools=mcp_tools)
graph.add_node("tool_node", common_task_tool_node)

# Add edges
graph.add_edge(START, "task_classification")
graph.add_edge("task_classification", "summarization")
graph.add_edge("summarization", "llm_call")
graph.add_edge("human_node", "summarization")
graph.add_edge("tool_node", "summarization")

# Add conditional edges - Send objects are handled automatically
graph.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "tool_node": "tool_node",
        "ask_human": "human_node",
        "polish_response": "polish_response"
    }
)

common_task_agent = graph.compile()