"""Dataset builder for final-answer evaluations."""

from __future__ import annotations

from typing import Sequence

from dotenv import load_dotenv
from langsmith import Client

FINAL_RESULT_DATASET = "Daily Chat DataSet"

EXAMPLES: Sequence[dict] = (
    {
        "inputs": {"question": "今天天气如何"},
        "outputs": {"answer": "当前温度 25° · 部分晴"},
    },
    {
        "inputs": {"question": "ChatGpt的CodeX是什么"},
        "outputs": {
            "answer": (
                "CodeX（也常写作 Codex）是 OpenAI 在 GPT-3 基础上专门针对编程任务训练的模型。"
                "它是 GitHub Copilot 背后的核心模型。简单来说，Codex 可以理解自然语言并生成代码，"
                "或者阅读代码并解释其含义"
            ),
        },
    },
)


def create_final_result_dataset(dataset_name: str = FINAL_RESULT_DATASET) -> str:
    """Create the LangSmith dataset used for final-answer grading."""
    load_dotenv()
    client = Client()
    dataset = client.create_dataset(dataset_name)
    client.create_examples(dataset_id=dataset.id, examples=list(EXAMPLES))
    return dataset.id


if __name__ == "__main__":
    dataset_id = create_final_result_dataset()
    print(f"Created dataset '{FINAL_RESULT_DATASET}' with id: {dataset_id}")
