"""Evaluate the agent execution trajectory against LangSmith references."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
load_dotenv()
from langchain_core.messages import HumanMessage
from langsmith import Client

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from assistant_agent.daily_chat_agent import daily_chatter_agent  # noqa: E402
from assistant_agent.states.state_daily_chatter import (  # noqa: E402
    DailyChatterAgentState,
)


DATASET_NAME = "Chat FINIAL & TRACE DataSet"
_client = Client()


def trajectory_subsequence(outputs: dict, reference_outputs: dict) -> float:
    """Return the longest matching subsequence ratio between actual and reference trajectories."""
    actual = outputs.get("trajectory", [])
    expected = reference_outputs.get("trajectory", [])
    if not expected:
        return 0.0
    i = j = 0
    while i < len(actual) and j < len(expected):
        if actual[i] == expected[j]:
            j += 1
        i += 1
    return j / len(expected)


async def collect_trajectory(inputs: dict) -> Dict[str, List[str]]:
    """Stream the agent execution and record node + tool transitions."""
    trajectory: List[str] = []
    state = DailyChatterAgentState(
        messages=[HumanMessage(content=inputs["question"])],
        user_info=inputs.get("user_info"),
        tool_call_iterations=inputs.get("tool_call_iterations", 0),
    )
    async for _, event in daily_chatter_agent.astream(
        state, subgraphs=True, stream_mode="debug"
    ):
        if event.get("type") != "task":
            continue
        name = event.get("payload", {}).get("name")
        if name:
            trajectory.append(name)
        if name == "tool_node":
            tool_calls = event.get("payload", {}).get("input", {}).get("messages", [])
            if tool_calls:
                last_message = tool_calls[-1]
                for call in getattr(last_message, "tool_calls", []) or []:
                    tool_name = call.get("name")
                    if tool_name:
                        trajectory.append(tool_name)
    return {"trajectory": trajectory}


async def main() -> None:
    """Run trajectory comparison against the LangSmith dataset."""
    experiment = await _client.aevaluate(
        collect_trajectory,
        data=DATASET_NAME,
        evaluators=[trajectory_subsequence],
        experiment_prefix="daily-chat-trajectory",
    )
    print("Started experiment:", experiment)
    if hasattr(experiment, "to_pandas"):
        print(experiment.to_pandas())


if __name__ == "__main__":
    asyncio.run(main())
