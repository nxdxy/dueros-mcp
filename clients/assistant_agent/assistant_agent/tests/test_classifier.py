"""
分类器测试数据集和评估脚本 - 使用LangSmith

此模块使用LangSmith创建测试数据集并评估任务分类器的准确性。
使用精确匹配的评估方式来评估分类的准确性。
"""

import asyncio
import os
from typing import List, Dict, Any
from dotenv import load_dotenv

# 在所有其他导入之前加载环境变量
load_dotenv()

from langsmith import Client
from langsmith.evaluation import aevaluate
from langchain_core.messages import HumanMessage
from assistant_agent.classifier import classify_task
from assistant_agent.states.state_classifier import AgentInputState

# 初始化LangSmith客户端
client = Client()

# 测试数据集：包含10个测试用例
TEST_EXAMPLES = [
    {
        "inputs": {"message": "小度小度，播放周杰伦的歌"},
        "outputs": {"expected_task": "xiaodu_assistant"},
        "metadata": {"description": "控制小度设备播放音乐"}
    },
    {
        "inputs": {"message": "今天北京的天气怎么样？"},
        "outputs": {"expected_task": "daily_chatter"},
        "metadata": {"description": "查询天气信息"}
    },
    {
        "inputs": {"message": "你好小度，我们聊聊天吧"},
        "outputs": {"expected_task": "daily_chatter"},
        "metadata": {"description": "日常聊天对话"}
    },
    {
        "inputs": {"message": "小度音箱怎么连接WiFi？"},
        "outputs": {"expected_task": "xiaodu_assistant"},
        "metadata": {"description": "小度设备使用说明"}
    },
    {
        "inputs": {"message": "帮我给卧室音箱设置明天早上8点的闹钟"},
        "outputs": {"expected_task": "xiaodu_assistant"},
        "metadata": {"description": "设置音箱闹钟"}
    },
    {
        "inputs": {"message": "你觉得今天的天气适合出门吗？"},
        "outputs": {"expected_task": "daily_chatter"},
        "metadata": {"description": "关于天气的聊天"}
    },
    {
        "inputs": {"message": "小度设备突然没声音了，怎么办？"},
        "outputs": {"expected_task": "xiaodu_assistant"},
        "metadata": {"description": "小度设备故障排除"}
    },
    {
        "inputs": {"message": "查询一下最新的科技新闻"},
        "outputs": {"expected_task": "daily_chatter"},
        "metadata": {"description": "查询新闻信息"}
    },
    {
        "inputs": {"message": "小度，你今天心情怎么样？"},
        "outputs": {"expected_task": "daily_chatter"},
        "metadata": {"description": "情感交流聊天"}
    },
    {
        "inputs": {"message": "asdfgh qwerty 123456"},
        "outputs": {"expected_task": "no_task"},
        "metadata": {"description": "无意义的输入"}
    }
]

async def classifier_function(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    分类器函数，用于LangSmith评估
    
    Args:
        inputs: 包含用户消息的字典
        
    Returns:
        包含预测任务的字典
    """
    try:
        # 构建输入状态
        input_state = AgentInputState(
            messages=[HumanMessage(content=inputs["message"])]
        )
        
        # 调用分类器
        result = await classify_task(input_state)
        
        return {
            "predicted_task": result["task"],
            "revised_request": result.get("revised_request", ""),
            "raw_request": result.get("raw_request", "")
        }
        
    except Exception as e:
        return {
            "predicted_task": "ERROR",
            "error": str(e)
        }

def exact_match_evaluator(run, example) -> Dict[str, Any]:
    """
    精确匹配评估器
    
    Args:
        run: LangSmith运行结果
        example: 测试样例
        
    Returns:
        评估结果
    """
    predicted_task = run.outputs.get("predicted_task", "")
    expected_task = example.outputs.get("expected_task", "")
    
    is_correct = predicted_task == expected_task
    
    return {
        "key": "exact_match",
        "score": 1.0 if is_correct else 0.0,
        "comment": f"Expected: {expected_task}, Predicted: {predicted_task}"
    }

def create_dataset(dataset_name: str = "Classifier_Test_Dataset") -> str:
    """
    在LangSmith中创建测试数据集
    
    Args:
        dataset_name: 数据集名称
        
    Returns:
        数据集ID
    """
    try:
        # 尝试获取现有数据集
        datasets = list(client.list_datasets(dataset_name=dataset_name))
        if datasets:
            dataset = datasets[0]
            print(f"使用现有数据集: {dataset_name} (ID: {dataset.id})")
        else:
            # 创建新数据集
            dataset = client.create_dataset(
                dataset_name=dataset_name,
                description="分类器测试数据集，包含10个测试用例用于评估任务分类准确性"
            )
            print(f"创建新数据集: {dataset_name} (ID: {dataset.id})")
        
        # 添加测试样例
        client.create_examples(
            dataset_id=dataset.id,
            examples=TEST_EXAMPLES
        )
        
        print(f"已添加 {len(TEST_EXAMPLES)} 个测试样例到数据集")
        return dataset.id
        
    except Exception as e:
        print(f"创建数据集时出错: {e}")
        raise

async def run_evaluation(dataset_name: str = "Classifier_Test_Dataset"):
    """
    运行LangSmith评估
    
    Args:
        dataset_name: 数据集名称
    """
    print("开始使用LangSmith评估分类器性能...")
    print("=" * 60)
    
    try:
        # 创建或获取数据集
        # dataset_id = create_dataset(dataset_name)
        
        # 运行评估
        results = await aevaluate(
            classifier_function,
            data=dataset_name,
            evaluators=[exact_match_evaluator],
            experiment_prefix="classifier_evaluation",
            description="分类器精确匹配评估",
            num_repetitions=1
        )
        
        print("\n评估完成!")
        print(f"实验名称: {results.experiment_name}")
        
        # 调试：打印results对象的属性
        print("Available attributes:", [attr for attr in dir(results) if not attr.startswith('_')])
        
        # 尝试获取样例数量
        example_count = 0
        if hasattr(results, 'results') and results.results:
            example_count = len(results.results)
        elif hasattr(results, 'runs') and results.runs:
            example_count = len(results.runs)
        
        if example_count > 0:
            print(f"总样例数: {example_count}")
            
            # 计算准确率
            correct_count = 0
            if hasattr(results, 'results'):
                for r in results.results:
                    if hasattr(r, 'evaluation_results') and r.evaluation_results.get('exact_match', {}).get('score', 0) == 1.0:
                        correct_count += 1
            elif hasattr(results, 'runs'):
                for r in results.runs:
                    if hasattr(r, 'evaluation_results') and r.evaluation_results.get('exact_match', {}).get('score', 0) == 1.0:
                        correct_count += 1
            
            accuracy = correct_count / example_count
            print(f"准确率: {accuracy:.2%} ({correct_count}/{example_count})")
        else:
            print("未找到评估结果")
        
        print(f"\n查看详细结果: https://smith.langchain.com/projects/p/{results.experiment_name}")
        
        return results
        
    except Exception as e:
        print(f"评估过程中出错: {e}")
        raise

if __name__ == "__main__":
    # 检查环境变量
    load_dotenv(dotenv_path="/Users/miaojuanjuan/Documents/项目/人工智能项目/MCP/baidu/personal-code/assistant-agent/.env")
    if not os.getenv("LANGSMITH_API_KEY"):
        print("警告: 未设置LANGSMITH_API_KEY环境变量")
        print("请设置LangSmith API密钥以使用评估功能")
    else:
        # 运行完整的评估流程
        asyncio.run(run_evaluation())