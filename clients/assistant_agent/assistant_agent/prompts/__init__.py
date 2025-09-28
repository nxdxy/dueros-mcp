"""
Prompts package for assistant agent.

This package contains all prompt templates used in the assistant agent system.
"""

from .prompts import (
    task_classification_prompt,
    xiaodu_assistant_execution_prompt,
    daily_chatter_execution_prompt
)
from .common_task_prompts import (
    common_task_execution_prompt
)

__all__ = [
    "task_classification_prompt",
    "xiaodu_assistant_execution_prompt", 
    "daily_chatter_execution_prompt",
    "common_task_execution_prompt"
]
