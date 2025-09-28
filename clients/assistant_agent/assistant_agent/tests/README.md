# 分类器评估工具

本目录包含用于评估任务分类器准确性的测试工具，使用LangSmith进行评估。

## 文件说明

- `test_classifier.py`: 主要的评估脚本，包含测试数据集和评估逻辑

## 功能特性

### 测试数据集
包含10个精心设计的测试用例，覆盖所有4个任务类别：

1. **xiaodu_assistant** (3个用例)
   - 控制小度设备播放音乐
   - 小度设备使用说明
   - 小度设备故障排除

2. **common_task_agent** (3个用例)
   - 查询天气信息
   - 设置提醒任务
   - 查询新闻信息

3. **daily_chatter** (3个用例)
   - 日常聊天对话
   - 关于天气的聊天
   - 情感交流聊天

4. **no_task** (1个用例)
   - 无意义的输入

### 评估方法
使用**精确匹配**的评估方式：
- 预测的任务类别必须与期望的任务类别完全一致才算正确
- 计算整体准确率和各任务类别的准确率

## 使用方法

### 1. 环境准备

首先需要设置LangSmith API密钥：

```bash
export LANGCHAIN_API_KEY="your_langsmith_api_key"
```

### 2. 安装依赖

确保已安装必要的依赖包：

```bash
pip install langsmith langchain-core
```

### 3. 运行评估

```bash
# 在项目根目录下运行
python -m assistant_agent.tests.test_classifier
```

### 4. 查看结果

评估完成后，可以通过以下方式查看结果：

1. **控制台输出**: 显示基本的准确率统计信息
2. **LangSmith Web界面**: 提供详细的评估结果和可视化分析

## 评估结果解读

评估将输出以下信息：

- **总样例数**: 测试用例的总数量
- **准确率**: 精确匹配的准确率百分比
- **实验名称**: LangSmith中的实验名称
- **详细结果链接**: 指向LangSmith Web界面的链接

## 自定义评估

### 修改测试数据集

可以通过修改 `TEST_EXAMPLES` 列表来添加或修改测试用例：

```python
TEST_EXAMPLES = [
    {
        "inputs": {"message": "你的测试消息"},
        "outputs": {"expected_task": "期望的任务类别"},
        "metadata": {"description": "测试用例描述"}
    },
    # ... 更多测试用例
]
```

### 添加新的评估器

可以创建自定义评估器来评估其他指标：

```python
def custom_evaluator(run, example) -> Dict[str, Any]:
    # 自定义评估逻辑
    return {
        "key": "custom_metric",
        "score": score,
        "comment": "评估说明"
    }
```

## 注意事项

1. 确保已正确配置LangSmith API密钥
2. 评估过程需要网络连接到LangSmith服务
3. 首次运行会创建新的数据集，后续运行会使用现有数据集
4. 每次评估会创建新的实验记录，便于对比不同版本的性能
