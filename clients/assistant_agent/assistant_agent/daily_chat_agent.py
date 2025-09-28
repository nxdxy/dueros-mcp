"""
Daily Chatter Agent

This agent is responsible for:
1. Chatting with user about daily life.
2. Calling relevant tools when users request functional controls, command executions, or real-time information like weather, stocks, news.
"""

from re import A
from assistant_agent.states.state_daily_chatter import DailyChatterAgentState, DailyChatterAgentOutputState
from assistant_agent.prompts.prompts import daily_chatter_execution_prompt, daily_chatter_hard_limit
from assistant_agent.utils import load_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage
from typing_extensions import Literal
from assistant_agent.utils import get_today_str
from assistant_agent.tools.mcp_manager import get_mcp_tools

# ===== CONFIGURATION =====
llm = load_chat_model("openrouter/anthropic/claude-sonnet-4")

# ===== AGENT NODES =====
async def llm_call(state: DailyChatterAgentState):
    """Analyze current state and decide on tool usage with MCP integration.

    This node:
    1. Retrieves available tools from MCP server
    2. Binds tools to the language model
    3. Processes user input and decides on tool usage

    Returns updated state with model response.
    """
    user_info = state.get("user_info")
    user_name_v = user_info.get("user_name", "用户") if user_info else "用户"

    prompt = [SystemMessage(content = daily_chatter_execution_prompt.format(date=get_today_str(), user_name = user_name_v))] + \
            state.get("messages", []) +  [HumanMessage(content = daily_chatter_hard_limit)]
    mcp_tools = await get_mcp_tools()


    model_with_tools = llm.bind_tools(mcp_tools)
    messages = await model_with_tools.ainvoke(prompt)
    return {
        "messages": messages
    }


async def tool_node(state: DailyChatterAgentState):
    """Call the tool based on the tool name and arguments.

    This node:
    1. Retrieves current tool calls from the last message
    2. Executes all tool calls using async operations (required for MCP)
    3. Returns formatted tool results
    """
    tool_calls = state["messages"][-1].tool_calls
    formatted_results = []
 
    async def execute_tools():
        mcp_tools = await get_mcp_tools()
        tool_map = {tool.name: tool for tool in mcp_tools}
        for tool_call in tool_calls:
            tool = tool_map[tool_call["name"]]
            tool_arguments = tool_call["args"]
            result = await tool.ainvoke(tool_arguments)
            formatted_results.append(result)

        tool_outputs = [
            ToolMessage(
                content=result,
                name=tool_call["name"],
                tool_call_id=tool_call["id"]
            )
            for tool_call, result in zip(tool_calls, formatted_results)
        ]
        return tool_outputs
    
    # Execute tools
    tool_outputs = await execute_tools()
    return {
        "messages": tool_outputs
    }


# ===== ROUTING LOGIC =====
def should_continue(state: DailyChatterAgentState) -> Literal["tool_node"]:
    """Determine if the workflow should continue.

    This function:
    1. Checks if the last message is a tool message
    2. Returns True if the workflow should continue, False otherwise
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tool_node"
    else:
        return END
    
# ===== GRAPH CONSTRUCTION =====
graph = StateGraph(DailyChatterAgentState, output_schema=DailyChatterAgentOutputState)
graph.add_node("llm_call", llm_call)
graph.add_node("tool_node", tool_node)
graph.add_edge(START, "llm_call")
graph.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "tool_node": "tool_node",
        END: END
    }
)
graph.add_edge("tool_node", "llm_call")

daily_chatter_agent = graph.compile()