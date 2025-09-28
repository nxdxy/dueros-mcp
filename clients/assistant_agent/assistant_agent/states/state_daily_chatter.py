"""

State Definitions and Pydantic Schemas for Daily Chatter Agent.

This module defines the state objects and structured schemas used for
the daily chatter agent workflow, including state management and output schemas.
"""

from typing_extensions import Optional, Annotated, Sequence
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from assistant_agent.utils import UserInfo

# ===== STATE DEFINITIONS =====
class DailyChatterAgentState(MessagesState):
    """
    State for the daily chatter agent.
    This state tracks the messages exchanged with daily chatter agent.
    """
    user_info: Optional[UserInfo]
    tool_call_iterations: int

# ===== STRUCTURED OUTPUT SCHEMAS =====
class DailyChatterAgentOutputState(MessagesState):
    """
    Output state for the daily chatter agent.
    This state tracks the messages exchanged with daily chatter agent.
    """