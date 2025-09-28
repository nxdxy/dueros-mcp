"""Evaluate a single planning step emitted by the daily chat agent."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv
load_dotenv()
from langsmith import Client

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from assistant_agent.daily_chat_agent import daily_chatter_agent  # noqa: E402
from assistant_agent.states.state_daily_chatter import DailyChatterAgentState  # noqa: E402

DATASET_NAME = "Chat Next Step"
_client = Client()


def evaluate_next_step(outputs: dict, reference_outputs: dict) -> Dict[str, object]:
    """Score whether the agent chose the expected number of tool calls."""
    message = outputs.get("messages")
    tool_calls = getattr(message, "tool_calls", []) or []
    expected = reference_outputs.get("call_tool_cnt", 0)
    return {
        "key": "correct_call_tools",
        "score": len(tool_calls) == expected,
        "observed_tool_calls": len(tool_calls),
        "expected_tool_calls": expected,
    }


async def collect_single_step(inputs: Dict[str, object]) -> Dict[str, object]:
    """Run only the first LLM planning step for evaluation."""
    state = DailyChatterAgentState(
        messages=inputs["messages"],
        user_info=inputs.get("user_info"),
        tool_call_iterations=inputs.get("tool_call_iterations", 0),
    )
    return await daily_chatter_agent.nodes["llm_call"].ainvoke(state)


async def main() -> None:
    """Evaluate the next-step planning behavior via LangSmith."""
    experiment = await _client.aevaluate(
        collect_single_step,
        data=DATASET_NAME,
        evaluators=[evaluate_next_step],
        experiment_prefix="daily-chat-next-step",
    )
    print("Started experiment:", experiment)
    if hasattr(experiment, "to_pandas"):
        print(experiment.to_pandas())


if __name__ == "__main__":
    asyncio.run(main())
