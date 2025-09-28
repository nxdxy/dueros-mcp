
"""
State Definitions and Pydantic Schemas for Request Classifying.

This module defines the state objects and structured schemas used for
the agent classifying workflow, including state management and output schemas.
"""

from pydantic import BaseModel, Field
from typing_extensions import Optional, Annotated, List, Sequence
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
import operator
from typing_extensions import Literal

# ===== STATE DEFINITIONS =====
class AgentInputState(MessagesState):
    """Input state for the full agent - only contains messages from user input."""
    pass

class AgentState(MessagesState):
    """
    Main state for the full multi-agent assistant system.

    Extends MessagesState with additional fields for assistant coordination.
    Note: Some fields are duplicated across different state classes for proper
    state management between subgraphs and the main workflow.
    """
    # Task generated from user request
    task: str
    # Revised request generated from user request
    revised_request: Optional[str]
    # Device info provided by user
    device_info: Optional[str]
    raw_request: str

# ===== STRUCTURED OUTPUT SCHEMAS =====
class TaskClassificationResult(BaseModel):
    """Schema for research brief generation."""
    task: Literal["xiaodu_assistant", "common_task_agent", "daily_chatter", "no_task"] = Field(
        description="A task that should be executed.",
    )
    revised_request: str = Field(
        description="A revised request that should be executed.",
    )
    raw_request: str = Field(
        description="The original user request.",
    )