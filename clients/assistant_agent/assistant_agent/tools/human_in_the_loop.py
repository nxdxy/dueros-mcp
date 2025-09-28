"""
Human-in-the-loop mechanism for tools

Provides a wrapper function to add human confirmation before tool execution.
"""

from typing import Callable, Optional, Dict, Any
from langchain_core.tools import BaseTool
from langchain_core.tools import tool as create_tool
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt
from langgraph.prebuilt.interrupt import HumanInterrupt, HumanInterruptConfig


def add_human_in_the_loop(
    tool: Callable | BaseTool,
    *,
    interrupt_config: Optional[HumanInterruptConfig] = None,
) -> BaseTool:
    """Wrap a tool to support human-in-the-loop review."""

    
    if not isinstance(tool, BaseTool):
        tool = create_tool(tool)

    if interrupt_config is None:
        interrupt_config = HumanInterruptConfig(
            allow_accept=True,
            allow_edit=False,
            allow_respond=True,
        )

    @create_tool(  
        tool.name,
        description=tool.description,
        args_schema=tool.args_schema
    )
    async def call_tool_with_interrupt(config: RunnableConfig = None, **tool_input):
        request: HumanInterrupt = {
            "action_request": {
                "action": tool.name,
                "args": tool_input
            },
            "config": interrupt_config,
            "description": f"Please review the tool call for {tool.name}"
        }
        response = interrupt([request])[0]
        # approve the tool call
        if response["type"] == "accept":
            if hasattr(tool, 'ainvoke'):
                tool_response = await tool.ainvoke(tool_input, config)
            else:
                tool_response = tool.invoke(tool_input, config)
        # update tool call args
        elif response["type"] == "edit":
            tool_input = response["args"]["args"]
            if hasattr(tool, 'ainvoke'):
                tool_response = await tool.ainvoke(tool_input, config)
            else:
                tool_response = tool.invoke(tool_input, config)
        # respond to the LLM with user feedback
        elif response["type"] == "response":
            user_feedback = response["args"]
            tool_response = user_feedback
        else:
            raise ValueError(f"Unsupported interrupt response type: {response['type']}")

        return tool_response

    return call_tool_with_interrupt