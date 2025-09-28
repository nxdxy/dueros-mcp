"""用于多智能体间交接（handoff）的工具。"""

from typing import Annotated, Optional
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from assistant_agent.states.state_daily_chatter import DailyChatterAgentState, DailyChatterAgentOutputState


def create_handoff_tool(*, agent_name: str, description: str | None = None):
    """创建 handoff 工具，允许代理主动转移到其他代理。"""
    name = f"transfer_to_{agent_name}"
    description = description or f"Transfer to {agent_name} for specialized assistance."

    @tool(name, description=description)
    def handoff_tool(
        state: Annotated[DailyChatterAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        reason: Optional[str] = None,
    ) -> Command:
        """handoff 工具实现。"""
        # 验证状态
        if not hasattr(state, 'messages') or state.messages is None:
            raise ValueError("Invalid state: messages attribute is missing or None")
        
        # 创建工具消息
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}" + (f" - {reason}" if reason else ""),
            "name": name,
            "tool_call_id": tool_call_id,
        }
        
        # 根据agent_name映射到严格的agent类型
        agent_type_mapping = {
            "common_task_agent_subgraph": "common_task_agent",
            "xiaodu_assistant_agent_subgraph": "xiaodu_assistant",
            "daily_chatter_agent_subgraph": "daily_chatter"
        }
        
        # 验证agent_name是否在映射中
        if agent_name not in agent_type_mapping:
            raise ValueError(f"Unknown agent_name: {agent_name}. Valid options: {list(agent_type_mapping.keys())}")
        
        agent_type = agent_type_mapping[agent_name]
        
        # 返回Command，包含状态更新和路由信息
        return Command(
            goto=agent_name,  # 目标agent
            update={
                "messages": state.messages + [tool_message],
                "last_active_agent": agent_name,
                "agent_type": agent_type,  # 使用映射后的严格agent类型
            },
            graph=Command.PARENT,  # 导航到父图
        )
    
    return handoff_tool


transfer_to_xiaodu_assistant = create_handoff_tool(
    agent_name="xiaodu_assistant_agent_subgraph",
    description="Transfer to the xiaodu assistant for xiaodu device-related issues, customer service, and device control functions."
)

# 所有handoff工具列表
HANDOFF_TOOLS = [
    transfer_to_xiaodu_assistant
] 