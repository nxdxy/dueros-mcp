"""使用 invoke 接口验证 common_task_agent 的 Human-in-the-loop 行为。"""

import asyncio
import importlib
import pathlib
import sys
import types
import uuid

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.types import Command


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _ensure_stub_modules() -> None:
    if "langchain_openai" not in sys.modules:
        module = types.ModuleType("langchain_openai")

        class _StubChatOpenAI:
            def __init__(self, *args, **kwargs):
                pass

            def bind(self, *args, **kwargs):
                return self

            def with_structured_output(self, *args, **kwargs):
                return self

            async def ainvoke(self, *_args, **_kwargs):
                return {}

        module.ChatOpenAI = _StubChatOpenAI
        sys.modules["langchain_openai"] = module

    if "langchain_anthropic" not in sys.modules:
        module = types.ModuleType("langchain_anthropic")

        class _StubChatAnthropic:
            def __init__(self, *args, **kwargs):
                pass

        module.ChatAnthropic = _StubChatAnthropic
        sys.modules["langchain_anthropic"] = module

    if "langmem" not in sys.modules:
        langmem_module = types.ModuleType("langmem")
        sys.modules["langmem"] = langmem_module

    if "langmem.short_term" not in sys.modules:
        short_term_module = types.ModuleType("langmem.short_term")

        class _SummarizationNode:
            def __init__(self, *args, **kwargs):
                pass

            async def __call__(self, state):
                return {}

        short_term_module.SummarizationNode = _SummarizationNode
        sys.modules["langmem.short_term"] = short_term_module
        sys.modules["langmem"].short_term = short_term_module  # type: ignore[attr-defined]

    if "langchain_mcp_adapters" not in sys.modules:
        parent = types.ModuleType("langchain_mcp_adapters")
        sys.modules["langchain_mcp_adapters"] = parent

    if "langchain_mcp_adapters.client" not in sys.modules:
        client_module = types.ModuleType("langchain_mcp_adapters.client")

        class _MultiServerMCPClient:
            def __init__(self, *args, **kwargs):
                pass

            async def get_tools(self):
                return []

        client_module.MultiServerMCPClient = _MultiServerMCPClient
        sys.modules["langchain_mcp_adapters.client"] = client_module
        sys.modules["langchain_mcp_adapters"].client = client_module  # type: ignore[attr-defined]


_ensure_stub_modules()


import assistant_agent.common_task_agent as cta
import assistant_agent.tools.mcp_manager as mcp_manager


class _FakeSummaryModel:
    def bind(self, *args, **kwargs):
        return self

    async def ainvoke(self, *_args, **_kwargs):
        return {}


class _FakeClassifierModel:
    def with_structured_output(self, schema):
        class _Structured:
            async def ainvoke(self, *_args, **_kwargs):
                return schema(task_complexity="simple", classification_reason="stub")

        return _Structured()


class _FakeSimpleModel:
    def __init__(self):
        self._calls = 0

    def bind_tools(self, _tools):
        parent = self

        class _Bound:
            async def ainvoke(self, *_args, **_kwargs):
                parent._calls += 1
                if parent._calls == 1:
                    return AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "AskHuman",
                                "args": {"question": "请问你所在的城市是哪里？"},
                                "id": "ask-human-1",
                            }
                        ],
                    )
                return AIMessage(content="北京今日天气晴，最高气温25℃。", tool_calls=[])

        return _Bound()


class _FakeComplexModel:
    async def ainvoke(self, *_args, **_kwargs):
        return "北京今日天气晴，最高气温25℃。"


cta.summary_llm = _FakeSummaryModel()
cta.classifier_llm = _FakeClassifierModel()
cta.simple_task_llm = _FakeSimpleModel()
cta.complex_task_llm = _FakeComplexModel()
cta.summarization_node = None  # 在同步图中自定义


async def _fake_get_mcp_tools():
    return []


def _fake_get_mcp_tools_sync():
    return []


mcp_manager.get_mcp_tools = _fake_get_mcp_tools  # type: ignore[assignment]
mcp_manager.get_mcp_tools_sync = _fake_get_mcp_tools_sync  # type: ignore[assignment]
cta.get_mcp_tools = _fake_get_mcp_tools  # type: ignore[attr-defined]
cta.get_mcp_tools_sync = _fake_get_mcp_tools_sync  # type: ignore[attr-defined]

ORIGINAL_TASK_CLASSIFICATION = cta.task_classification
ORIGINAL_LLM_CALL = cta.llm_call
ORIGINAL_HUMAN_NODE = cta.human_node
ORIGINAL_POLISH_RESPONSE = cta.polish_response


def _syncify(async_fn):
    async def _wrapper(state):
        return await async_fn(state)

    def _sync(state):
        return asyncio.run(_wrapper(state))

    return _sync


def _sync_summarization(state):
    messages = state.get("messages") if isinstance(state, dict) else getattr(state, "messages", [])
    return {
        "context": state.get("context", {}) if isinstance(state, dict) else {},
        "llm_input_messages": messages,
    }


def _build_sync_agent():
    graph_builder = StateGraph(cta.CommonTaskState, output_schema=cta.CommonTaskOutputState)
    graph_builder.add_node("task_classification", _syncify(ORIGINAL_TASK_CLASSIFICATION))
    graph_builder.add_node("summarization", _sync_summarization)
    graph_builder.add_node("llm_call", _syncify(ORIGINAL_LLM_CALL))
    graph_builder.add_node("human_node", _syncify(ORIGINAL_HUMAN_NODE))
    graph_builder.add_node("polish_response", _syncify(ORIGINAL_POLISH_RESPONSE))
    graph_builder.add_node("tool_node", ToolNode(tools=[]))

    graph_builder.add_edge(START, "task_classification")
    graph_builder.add_edge("task_classification", "summarization")
    graph_builder.add_edge("summarization", "llm_call")
    graph_builder.add_edge("human_node", "summarization")
    graph_builder.add_edge("tool_node", "summarization")
    graph_builder.add_conditional_edges(
        "llm_call",
        cta.should_continue,
        {
            "tool_node": "tool_node",
            "ask_human": "human_node",
            "polish_response": "polish_response",
        },
    )

    return graph_builder.compile(checkpointer=InMemorySaver())


cta.common_task_agent = _build_sync_agent()


def run_demo() -> None:
    thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    initial_state = cta.CommonTaskState(
        messages=[
            HumanMessage(content="你好"),
            AIMessage(content="你好，有什么可以帮忙吗？"),
            HumanMessage(content="查询我所在城市的天气"),
        ],
        tool_call_iterations=0,
    )

    first_result = cta.common_task_agent.invoke(initial_state, config=thread_config)
    interrupts = first_result.get("__interrupt__")
    assert interrupts, "预期触发 AskHuman 中断"
    interrupt_value = getattr(interrupts[0].value, "question", interrupts[0].value)
    assert interrupt_value == "请问你所在的城市是哪里？"

    print("--------- FINISH INTERRUPT ------------")

    resumed_state = cta.common_task_agent.invoke(Command(resume="北京"), config=thread_config)
    final_messages = resumed_state["messages"]
    assert isinstance(final_messages[-1], AIMessage)
    assert "北京" in final_messages[-1].content
    assert resumed_state["user_response"] == "北京今日天气晴，最高气温25℃。"

    print("初始对话: 你好 -> 你好，有什么可以帮忙吗？ -> 查询我所在城市的天气")
    print("第一次调用触发的提问:", interrupt_value)
    print("最终答案:", resumed_state["user_response"])


if __name__ == "__main__":  # pragma: no cover
    run_demo()
