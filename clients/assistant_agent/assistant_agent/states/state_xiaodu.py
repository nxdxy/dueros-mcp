"""
State Definitions and Pydantic Schemas for Xiaodu Assistant Agent.

This module defines the state objects and structured schemas used for
the xiaodu assistant agent workflow, including state management and output schemas.
"""

from typing_extensions import Optional, Annotated, Sequence
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from typing import List

# ===== STATE DEFINITIONS =====
class XiaoduAgentState(MessagesState):
    """
    State for the xiaodu assistant agent.
    This state tracks the messages exchanged with xiaodu agent, the tool call iterations and the device info provided by user.
    """
    device_info: Optional[str]
    xiaodu_messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_call_iterations: int
    revised_request: str
    available_devices: Optional[List[str]]

class XiaoduAgentOutputState(MessagesState):
    """
    Output state for the xiaodu assistant agent.
    This state tracks the messages exchanged with xiaodu agent and the device info provided by user.
    """
    xiaodu_messages: Annotated[Sequence[BaseMessage], add_messages]
