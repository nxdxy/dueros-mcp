"""自定义工具集合，包含小度智能客服等工具。"""

import json
import asyncio
from typing import Optional, Dict, Any
from langchain_core.tools import tool
import aiohttp
import uuid


@tool
def xiaodu_customer_service(
    query: str
) -> str:
    """
    获取小度设备的使用说明、产品介绍等信息。
    
    Args:
        query: 用户查询内容
    
    Returns:
        str: 智能客服的回复内容
    """
    # 构建请求数据，随机生成logid
    logid = str(uuid.uuid4())
    payload = {
        "query": query,
        "agent_id": "your_agent_id",
        "logid": logid,
        "stream": False,
        "debug": False,
        "uid": "uid"
    }
    
    # 构建请求头
    headers = {
        "Content-Type": "application/json"
    }
    # 客服能力暂未加入小度Mcp工具集合，为了演示小姑此处通过Tool的方式调用小度客服Agent
    # 如果用户有需求可以在Git社区联系我们 将此功能收录到小度McpServer中
    base_url = "http://host:ip/agent"
    try:
        # 使用asyncio运行异步HTTP请求
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 如果事件循环正在运行，使用asyncio.create_task
            return loop.run_until_complete(_make_async_request(base_url, payload, headers))
        else:
            return loop.run_until_complete(_make_async_request(base_url, payload, headers))
    except RuntimeError:
        # 没有事件循环，创建一个新的
        return asyncio.run(_make_async_request(base_url, payload, headers))

async def _make_async_request(url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> str:
    """异步HTTP请求实现"""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    # 检查响应内容类型
                    content_type = response.headers.get('content-type', '').lower()
                    
                    if 'application/x-ndjson' in content_type or 'application/ndjson' in content_type:
                        # 处理 NDJSON 格式
                        text_content = await response.text()
                        lines = text_content.strip().split('\n')
                        results = []
                        
                        for line in lines:
                            if line.strip():  # 跳过空行
                                try:
                                    json_obj = json.loads(line)
                                    results.append(json_obj)
                                except json.JSONDecodeError:
                                    # 如果某行不是有效的JSON，直接添加为字符串
                                    results.append(line)
                        
                        # 合并所有结果
                        if len(results) == 1:
                            result = results[0]
                        else:
                            result = results
                    else:
                        # 处理标准 JSON 格式
                        result = await response.json()
                    
                    # 根据API响应格式提取回复内容
                    if isinstance(result, dict):
                        # 如果返回的是字典，尝试提取常见的回复字段
                        if "response" in result:
                            return result["response"]
                        elif "answer" in result:
                            return result["answer"]
                        elif "content" in result:
                            return result["content"]
                        elif "message" in result:
                            return result["message"]
                        else:
                            # 如果没有找到常见字段，返回整个响应的字符串表示
                            return json.dumps(result, ensure_ascii=False, indent=2)
                    elif isinstance(result, list):
                        # 如果是列表，尝试提取每个元素的回复内容
                        responses = []
                        for item in result:
                            if isinstance(item, dict):
                                if "response" in item:
                                    responses.append(item["response"])
                                elif "answer" in item:
                                    responses.append(item["answer"])
                                elif "content" in item:
                                    responses.append(item["content"])
                                elif "message" in item:
                                    responses.append(item["message"])
                                else:
                                    responses.append(json.dumps(item, ensure_ascii=False))
                            else:
                                responses.append(str(item))
                        return '\n'.join(responses)
                    else:
                        return str(result)
                else:
                    error_msg = f"API请求失败，状态码: {response.status}"
                    try:
                        error_detail = await response.text()
                        error_msg += f"，错误详情: {error_detail}"
                    except:
                        pass
                    return error_msg
        except aiohttp.ClientError as e:
            return f"网络请求错误: {str(e)}"
        except json.JSONDecodeError as e:
            return f"JSON解析错误: {str(e)}"
        except Exception as e:
            return f"未知错误: {str(e)}"

# 导出所有自定义工具
CUSTOM_TOOLS = [
    xiaodu_customer_service
]

