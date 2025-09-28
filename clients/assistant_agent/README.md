# 小度助手多智能体系统 (Assistant Agent)

基于 LangGraph 构建的小度多智能体助手，结合 MCP 工具体系与自定义服务，能够在一个工作流中完成任务分类、设备控制、日常对话以及通用事务处理。系统按照“感知→决策→执行”的流程组织，每个智能体专注于单一领域，并通过结构化状态在同一个图中协同。

## 系统概览

- **任务编排**：`assistant_agent/agent_full.py` 通过 `StateGraph` 将任务分类器与 3 个子智能体拼装成完整链路。
- **状态驱动**：各 Agent 的 `State` 与结构化输出集中在 `assistant_agent/states/`，保证上下文可追踪、易于扩展。
- **工具体系**：`assistant_agent/tools/mcp_manager.py` 统一接入 MCP Server；小度专属工具、人审包裹器和 handoff 工具进一步拓展能力。
- **LangGraph 生态**：项目根目录的 `langgraph.json` 定义图入口，可直接 `langgraph dev` 启动可视化调试。

## 核心能力

### 🤖 多智能体协作
- **任务分类** (`assistant_agent/classifier.py`)：调用 OpenRouter 模型产出结构化结果，决定路由目标与改写后的用户指令。
- **小度设备助手** (`assistant_agent/xiaodu_agent.py`)：汇总 MCP 工具与自定义客服工具，针对设备控制、资源推送等场景提供人审保护。
- **通用任务助手** (`assistant_agent/common_task_agent.py`)：根据复杂度在 GPT-4o / Claude 间切换，利用 `SummarizationNode` 控制上下文长度，并可通过 `AskHuman` 工具触发人工澄清。
- **日常闲聊助手** (`assistant_agent/daily_chat_agent.py`)：加载 MCP 工具完成天气、新闻等实时查询，维持人设化的对话体验。

### 🛠️ 工具与Human In The Loop
- **MCP 集成**：`mcp_config.json` 定义百度地图、AI 搜索、爱奇艺等服务，支持通过环境变量安全注入密钥。
- **自定义客服**：`assistant_agent/tools/custom_tools.py` 演示如何以异步 HTTP 接入小度客服（示例需要替换真实入口）。
- **人审拦截**：`assistant_agent/tools/human_in_the_loop.py` 利用 `langgraph` 的 `interrupt` 能力，对关键工具调用征询确认。
- **交接工具**：`assistant_agent/tools/handoff_tools.py` 允许 Agent 在图内显式跳转，便于扩展更多业务分支。

### 🧠 记忆与状态管理
- 通过 `MessagesState` 派生的状态类记录会话、用户画像、设备上下文。
- `common_task_agent` 结合 `SummarizationNode` 与自定义 `fixed_preprocess_messages.py`，确保长对话仍保留完整的工具调用序列。
- `xiaodu_agent` 在列出设备后支持 `interrupt` 选择，保证设备绑定的准确性。

## 目录速览

| 目录/文件 | 作用 |
| --- | --- |
| `assistant_agent/agent_full.py` | 多智能体主图入口 |
| `assistant_agent/states/` | 各 Agent 状态、结构化输出模型 |
| `assistant_agent/prompts/` | 任务分类、执行、人设等提示词 |
| `assistant_agent/tools/` | MCP 客户端、自定义工具、人审封装、handoff |
| `assistant_agent/tests/` | LangGraph human-in-loop 行为与分类器评测脚本 |
| `evaluate/` | 基于 LangSmith 的离线评估、数据集脚本 |
| `langgraph.json` | LangGraph CLI/Playground 配置 |
| `mcp_config.json` | MCP Server 配置模板 |

## 快速开始

### 1. 准备环境
```bash
conda create -n assistant-agent python=3.11
conda activate assistant-agent
pip install -e .[dev]
```
> 仅体验可使用 `pip install -e .`；开发建议带上 `[dev]` 以启用 Ruff、pytest 等工具。

### 2. 配置密钥
```bash
mcp_config.json              # 模版中替换自己的api-key
.env                         # 参考.env.example模版设置.env文件
```
- 在 `.env` 中填充 `OPENROUTER_API_KEY`、`LANGSMITH_API_KEY`、`TAVILY_API_KEY` 等模型与评估密钥。
- 更新 `mcp_config.json` 中的 `${your_api_key}`、`${your_access_token}` 占位符。

### 3. 启动 LangGraph 工作流
```bash
# 启动完整多智能体流程
langgraph dev
```
- `langgraph dev` 会读取 `langgraph.json`，可在 Playground 中调试 `agent_full` 或单独子图。

## 测试与评估

- **单测/回归**：
```bash
pytest assistant_agent/tests -k human_loop
pytest assistant_agent/tests -k classifier
```
  - `test_common_task_human_loop.py` 使用 stub 模型验证人审流程。
  - `test_classifier.py` 依赖 LangSmith 数据集与真实 LLM，请在 `.env` 中提供相关 API Key。
- **LangSmith 评估**：`evaluate/` 目录包含日常闲聊回复质量、下一步行动轨迹等脚本，例如：
```bash
python -m evaluate.final_result_evaluation
```
 运行前需在 LangSmith 控制台准备对应数据集名称（脚本默认示例：`Daily Chat DataSet`）。
