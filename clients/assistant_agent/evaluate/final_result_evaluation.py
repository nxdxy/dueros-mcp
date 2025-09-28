"""Evaluate final answers produced by the daily chat agent."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv
load_dotenv()
from langchain_core.messages import HumanMessage
from langsmith import Client

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from assistant_agent.daily_chat_agent import daily_chatter_agent
from assistant_agent.states.state_daily_chatter import DailyChatterAgentState
from assistant_agent.utils import load_chat_model


MODEL_NAME = "openrouter/openai/gpt-4o-mini"
DATASET_NAME = "Daily Chat DataSet"

_GRADER_INSTRUCTIONS = "你是一位专门负责为学生答题评分的专家教授"
_llm = load_chat_model(MODEL_NAME)
_client = Client()


def _build_user_prompt(inputs: Dict[str, str], outputs: Dict[str, str], reference: Dict[str, str]) -> str:
    return (
        "评估如下问题:\n"
        f"{inputs['question']}\n"
        "标准回答如下:\n"
        f"{reference['answer']}\n"
        "评估如下回答的准确性:\n"
        f"{outputs['response']}\n"
        "返回 CORRECT 或 INCORRECT:\n"
    )


async def correctness(inputs: Dict[str, str], outputs: Dict[str, str], reference_outputs: Dict[str, str]) -> bool:
    """Judge whether the agent response matches the reference answer."""
    user_prompt = _build_user_prompt(inputs, outputs, reference_outputs)
    response = await _llm.ainvoke(
        [
            {"role": "system", "content": _GRADER_INSTRUCTIONS},
            {"role": "user", "content": user_prompt},
        ]
    )
    content = getattr(response, "content", "")
    return str(content).strip().upper().startswith("CORRECT")


def concision(outputs: Dict[str, str], reference_outputs: Dict[str, str]) -> bool:
    """Return True if the reply is succinct relative to the reference answer."""
    return len(outputs.get("response", "")) < 2 * len(reference_outputs.get("answer", ""))


async def generate_response(inputs: Dict[str, str]) -> Dict[str, str]:
    """Invoke the daily chat agent and collect the final response."""
    state = DailyChatterAgentState(
        messages=[HumanMessage(content=inputs["question"])],
        user_info=inputs.get("user_info"),
        tool_call_iterations=inputs.get("tool_call_iterations", 0),
    )
    result_state = await daily_chatter_agent.ainvoke(state)
    message = result_state["messages"][-1]
    return {"response": getattr(message, "content", "")}


async def main() -> None:
    """Run the LangSmith hosted evaluation for final-answer quality."""
    experiment = await _client.aevaluate(
        generate_response,
        data=DATASET_NAME,
        evaluators=[concision, correctness],
        experiment_prefix="daily-chat-final",
    )
    print("Started experiment:", experiment)
    if hasattr(experiment, "to_pandas"):
        print(experiment.to_pandas())


if __name__ == "__main__":
    asyncio.run(main())
