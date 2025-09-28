"""
Xiaodu Assistant Agent

This agent is responsible for:
1. Providing information about xiaodu devices.
2. Control xiaodu's smart speaker, smart home devices with device info provided by user.

"""

from assistant_agent.utils import load_chat_model
from assistant_agent.states.state_xiaodu import XiaoduAgentState
from assistant_agent.prompts.prompts import xiaodu_assistant_execution_prompt
from assistant_agent.tools.mcp_manager import get_mcp_tools
from assistant_agent.tools.custom_tools import CUSTOM_TOOLS
from assistant_agent.tools.human_in_the_loop import add_human_in_the_loop
from langchain_core.messages import ToolMessage, SystemMessage, AIMessage
from typing_extensions import Literal
from langgraph.graph import StateGraph, START, END
import json
from langgraph.types import interrupt
from langgraph.prebuilt.interrupt import HumanInterruptConfig

# ===== CONFIGURATION =====
# Initialize model
llm = load_chat_model("openrouter/openai/gpt-4o")

# Tools that require human confirmation
HUMAN_CONFIRMATION_TOOLS = ["xiaodu_speak", "control_xiaodu", "push_resource_to_xiaodu", "xiaodu_take_photo"]

def apply_human_in_the_loop_to_tools(tools):
    """Apply human-in-the-loop mechanism to specified tools"""
    wrapped_tools = []
    for tool in tools:
        if tool.name in HUMAN_CONFIRMATION_TOOLS:
            # Create human-in-the-loop version with specific config for xiaodu tools
            interrupt_config = HumanInterruptConfig(
                allow_accept=True,
                allow_edit=False,
                allow_respond=True,
            )
            wrapped_tool = add_human_in_the_loop(tool, interrupt_config=interrupt_config)
            wrapped_tools.append(wrapped_tool)
        else:
            wrapped_tools.append(tool)
    return wrapped_tools

# ===== AGENT NODES =====
async def extract_device_info(state: XiaoduAgentState):
    """Interact with user to select a device."""
    available_devices = state.get("available_devices")
    query = f"请选择一个设备(编号1-{len(available_devices)}): {available_devices}"
    while True:
        answer = interrupt(query)
        try:
            answer = int(answer)
            if 1 <= answer <= len(available_devices):
                break
        except ValueError:
            query = f"'{answer}' 不是有效的设备编号，请输入1-{len(available_devices)}之间的数字: {available_devices}"

    return {
        "device_info": available_devices[answer - 1],
        "available_devices": available_devices
    }

async def llm_call(state: XiaoduAgentState):
    """Analyze current state and decide on tool usage with MCP integration.

    This node:
    1. Retrieves available tools from MCP server
    2. Binds tools to the language model
    3. Processes user input and decides on tool usage

    Returns updated state with model response.
    """
    # Get available tools from MCP server
    mcp_tools = await get_mcp_tools()

    # Combine MCP tools with custom tools
    all_tools = mcp_tools + CUSTOM_TOOLS 
    
    # Apply human-in-the-loop to specified tools
    tools = apply_human_in_the_loop_to_tools(all_tools)

    # Initialize model with tool binding
    llm_with_tools = llm.bind_tools(tools)
    response = await llm_with_tools.ainvoke([SystemMessage(content=xiaodu_assistant_execution_prompt.format(messages=state["messages"], user_request=state["revised_request"], device_info=state.get("device_info", ""), available_devices=state.get("available_devices", "")))])
    return {
        "messages": [response]
    }

# tool node
async def tool_node(state: XiaoduAgentState):
    """Call the tool based on the tool name and arguments.

    This node:
    1. Retrieves current tool calls from the last message
    2. Executes all tool calls using async operations (required for MCP)
    3. Returns formatted tool results

    Note: MCP requires async operations due to inter-process communication
    with the MCP server subprocess. This is unavoidable.
    """
    tool_calls = state["messages"][-1].tool_calls
    formatted_results = []
    
    # Get available tools from MCP server
    mcp_tools = await get_mcp_tools()
    
    # Combine MCP tools with custom tools
    combined_tools = mcp_tools + CUSTOM_TOOLS
    
    # Apply human-in-the-loop to specified tools
    all_tools = apply_human_in_the_loop_to_tools(combined_tools) 
        
    async def execute_tools():
        tool_map = {tool.name: tool for tool in all_tools}
        available_devices = state.get("available_devices", None)
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_arguments = tool_call["args"]
            result = await tool_map[tool_name].ainvoke(tool_arguments)
            formatted_results.append(result)
            if tool_name == "list_user_devices":
                devices_data = json.loads(result)
                if isinstance(devices_data, list):
                    available_devices = devices_data
                else:
                    available_devices = [devices_data]
                    
        # Format results as tool messages
        tool_outputs = [
            ToolMessage(
                content=result,
                name=tool_call["name"],
                tool_call_id=tool_call["id"]
            )
            for tool_call, result in zip(tool_calls, formatted_results)
        ]

        return tool_outputs, available_devices
    
    messages, available_devices = await execute_tools()
    
    return {
        "messages": messages,
        "available_devices": available_devices
    }

# ===== ROUTING LOGIC =====
def should_continue(state: XiaoduAgentState) -> Literal["tool_node"]:
    """Determine if the workflow should continue.

    This function:
    1. Checks if the last message is a tool message
    2. Returns True if the workflow should continue, False otherwise
    """
    last_message = state["messages"][-1]
    # 只有AIMessage才有tool_calls属性，ToolMessage没有
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tool_node"
    else:
        return END

# ===== GRAPH CONSTRUCTION =====
graph = StateGraph(XiaoduAgentState)
graph.add_node("llm_call", llm_call)
graph.add_node("tool_node", tool_node)
graph.add_node("extract_device_info", extract_device_info)
graph.add_edge(START, "llm_call")
graph.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "tool_node": "tool_node",
        END: END
    }
)

# 如果available_devices不为空且device_info为空，则提取设备信息
def extract_device_info_condition(state: XiaoduAgentState):
    if state.get("available_devices") is not None and state.get("device_info") is None:
        return "extract_device_info"
    else:
        return "llm_call"
    
    
graph.add_conditional_edges(
    "tool_node",
    extract_device_info_condition,
    {
        "extract_device_info": "extract_device_info",
        "llm_call": "llm_call"
    }
)
graph.add_edge("extract_device_info", "llm_call")

xiaodu_assistant_agent = graph.compile()