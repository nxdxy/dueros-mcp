"""Dataset builder for next-step (single iteration) evaluations."""

from __future__ import annotations

from typing import Sequence

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langsmith import Client

NEXT_STEP_DATASET = "Chat Next Step"

EXAMPLES: Sequence[dict] = (
    {
        "inputs": {
            "messages": [
                HumanMessage(content="最近基于游戏需求，装电脑用推荐什么显卡"),
            ]
        },
        "outputs": {"call_tool_cnt": 1},
    },
    {
        "inputs": {
            "messages": [
                HumanMessage(content="今天有什么有趣的新闻"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_AIsearch",
                            "args": {
                                "query": "今天 新闻 头条",
                                "instruction": "提供最新有趣的新闻摘要",
                            },
                            "name": "AIsearch",
                        }
                    ],
                ),
                ToolMessage(
                    content=(
                        "今日头条\nContent:  加沙战火持续美以“火上浇油”苹果黄牛“暴利”时代终结了吗"
                        "中国眼科“飞行医院”要来了俄方愿就乌危机有条件妥协记者:沙特王储赌赢了香港挖出一枚战时遗留炸弹"
                        "美宾夕法尼亚州发生枪击事件以色列是怎样筛选和招募间谍的贵州“城超”不一样iPhone17开售你会排队购买吗"
                        "加沙战火持续美以“火上浇油” 总书记治国理政故事"
                    ),
                    tool_call_id="call_AIsearch",
                    name="AIsearch",
                ),
            ]
        },
        "outputs": {"call_tool_cnt": 0},
    },
)


def create_next_step_dataset(dataset_name: str = NEXT_STEP_DATASET) -> str:
    """Create the LangSmith dataset used for next-step evaluation."""
    load_dotenv()
    client = Client()
    dataset = client.create_dataset(dataset_name)
    client.create_examples(dataset_id=dataset.id, examples=list(EXAMPLES))
    return dataset.id


if __name__ == "__main__":
    dataset_id = create_next_step_dataset()
    print(f"Created dataset '{NEXT_STEP_DATASET}' with id: {dataset_id}")
