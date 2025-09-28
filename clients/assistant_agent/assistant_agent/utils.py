"""
Utility functions for the assistant agent.
"""
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel
from pathlib import Path

class UserInfo(BaseModel):
    """User info"""
    user_name: str
    location: str

# ===== UTILITY FUNCTIONS =====

def get_current_dir() -> Path:
    """Get the current directory of the module.

    This function is compatible with Jupyter notebooks and regular Python scripts.

    Returns:
        Path object representing the current directory
    """
    try:
        return Path(__file__).resolve().parent
    except NameError:  # __file__ is not defined
        return Path.cwd()
    
def get_today_str() -> str:
    """Get current date in a human-readable format."""
    return datetime.now().strftime("%a %b %-d, %Y")

def load_chat_model(fully_specified_name: str):
    """根据完整名称加载聊天模型。

    Args:
        fully_specified_name: 形如 "provider/model-name" 的模型名

    Returns:
        已加载的聊天模型实例
    """
    if "/" not in fully_specified_name:
        raise ValueError(f"Model name must be in format 'provider/model-name', got: {fully_specified_name}")
    
    provider, model_name = fully_specified_name.split("/", 1)
    
    if provider == "openai":
        return ChatOpenAI(model=model_name)
    elif provider == "anthropic":
        return ChatAnthropic(model=model_name)
    elif provider == "openrouter":
        # OpenRouter 使用 OpenAI 兼容的 API
        import os
        return ChatOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            model=model_name,
            default_headers={
                "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL"),
                "X-Title": os.getenv("OPENROUTER_SITE_NAME"),
            }
        )
    else:
        raise RuntimeError(
            "No supported LLM provider available for the specified model. "
            "Please install appropriate provider packages or configure langchain_community."
        )