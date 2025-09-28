"""
State Definitions and Pydantic Schemas for Common Task Agent.

This module defines the state objects and structured schemas used for
the common task agent, including state management and output schemas.
"""

from typing_extensions import Optional
from langgraph.graph import MessagesState

from assistant_agent.utils import UserInfo
from pydantic import BaseModel, Field
from typing import Literal, List, Any, Dict
from langchain_core.messages import BaseMessage

# ===== STATE DEFINITIONS =====
class CommonTaskState(MessagesState):
    """
    State for the common task agent.
    """
    device_info: Optional[str]
    user_info: Optional[UserInfo]
    tool_call_iterations: int
    llm_input_messages: Optional[List[BaseMessage]] = Field(
        default=None,
        description="经过SummarizationNode处理后的消息列表，用于LLM输入"
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="SummarizationNode内部状态跟踪"
    )
    task_complexity: Optional[Literal["simple", "complex"]] = Field(
        default=None,
        description="任务复杂度分类：simple(普通任务) 或 complex(复杂任务)"
    )

class CommonTaskOutputState(MessagesState):
    """
    Output Schema for the common task agent.
    包含完整的messages字段和用于用户展示的response字段.
    """
    user_response: Optional[str] = Field(
        default=None, 
        description="用户展示的回复内容，从assistant消息中提取response部分，不包含internal_thought"
    )

# -------- STRUCTURED SCHEMA --------
class CommonTaskClassifation(BaseModel):
    """通用任务的难度分类"""

    task_complexity: Literal["simple", "complex"] = Field(
        description="任务的难易程度",
    )
    classification_reason: str = Field(
        description="分类的理由",
    )