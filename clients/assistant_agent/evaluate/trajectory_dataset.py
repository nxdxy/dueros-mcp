"""Dataset builder for trajectory-aware evaluations."""

from __future__ import annotations

from typing import Sequence

from dotenv import load_dotenv
from langsmith import Client

TRAJECTORY_DATASET = "Chat FINIAL & TRACE DataSet"

EXAMPLES: Sequence[dict] = (
    {
        "inputs": {"question": "最近基于游戏需求，装电脑用推荐什么显卡"},
        "outputs": {
            "answer": "AMD 9070Xt显卡",
            "trajectory": ["AIsearch"],
        },
    },
    {
        "inputs": {"question": "今天有什么有趣的新闻"},
        "outputs": {
            "answer": "美日举行史上规模极大的联合军演",
            "trajectory": ["AIsearch"],
        },
    },
)


def create_trajectory_dataset(dataset_name: str = TRAJECTORY_DATASET) -> str:
    """Create the LangSmith dataset used for trajectory scoring."""
    load_dotenv()
    client = Client()
    dataset = client.create_dataset(dataset_name)
    client.create_examples(dataset_id=dataset.id, examples=list(EXAMPLES))
    return dataset.id


if __name__ == "__main__":
    dataset_id = create_trajectory_dataset()
    print(f"Created dataset '{TRAJECTORY_DATASET}' with id: {dataset_id}")
