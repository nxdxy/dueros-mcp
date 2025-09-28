"""
MCP Client Manager

This class is responsible for managing the MCP client and the tools provided by the MCP server.
"""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool
import json

mcp_servers: Dict[str, Dict[str, Any]]

async def from_json_file(file_path: str) -> Dict[str, Dict[str, Any]]:
    """ Load MCP servers from json file """
    def _read_file():
        with open(file_path, "r") as f:
            return json.load(f)
    
    data = await asyncio.to_thread(_read_file)
    return data.get("mcpServers", {})

class MCPClientManager:
    """MCP Client Manager"""
    
    def __init__(self):
        self.config: Optional[Dict[str, Dict[str, Any]]] = None
        self._mcp_client: Optional[MultiServerMCPClient] = None
        self._tools: Optional[List[BaseTool]] = None
    
    def _build_mcp_servers(self) -> Dict[str, Dict[str, Any]]:
        """Build MCP servers with processed environment variables"""
        mcp_servers: Dict[str, Dict[str, Any]] = {}
        for server_name, server_config in self.config.items():
            processed_config = server_config.copy()
            # 处理headers中的环境变量占位符 ${VAR}
            if "headers" in processed_config:
                headers = processed_config["headers"].copy()
                for key, value in headers.items():
                    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                        env_var = value[2:-1]  # 移除 ${ 和 }
                        headers[key] = os.getenv(env_var, "")
                processed_config["headers"] = headers
            mcp_servers[server_name] = processed_config
        return mcp_servers
        
    async def _get_mcp_client(self) -> MultiServerMCPClient:
        """Get or create MCP client"""
        if self._mcp_client is None:
            # Load config if not already loaded
            if self.config is None:
                self.config = await from_json_file("mcp_config.json")
            
            # 处理环境变量替换
            mcp_servers = {}
            for server_name, server_config in self.config.items():
                processed_config = server_config.copy()
                
                # 处理headers中的环境变量
                if "headers" in processed_config:
                    headers = processed_config["headers"].copy()
                    for key, value in headers.items():
                        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                            env_var = value[2:-1]  # 移除 ${ 和 }
                            headers[key] = os.getenv(env_var, "")
                    processed_config["headers"] = headers
                
                mcp_servers[server_name] = processed_config
            # Create MultiServerMCPClient
            self._mcp_client = MultiServerMCPClient(mcp_servers)
            
        return self._mcp_client
    
    async def get_tools(self) -> List[BaseTool]:
        """Get all tools provided by MCP servers"""
        if self._tools is None:
            client = await self._get_mcp_client()
            self._tools = await client.get_tools()

        return self._tools
    
    async def close(self):
        """Close MCP client connection"""
        # MultiServerMCPClient 不需要显式关闭
        self._mcp_client = None
        self._tools = None

# 全局客户端管理器实例
_mcp_client_manager: Optional[MCPClientManager] = None


def get_mcp_client_manager() -> MCPClientManager:
    """Get MCP client manager instance"""
    global _mcp_client_manager
    if _mcp_client_manager is None:
        _mcp_client_manager = MCPClientManager()
    return _mcp_client_manager

async def get_mcp_tools() -> List[BaseTool]:
    """Get all MCP tools"""
    manager = get_mcp_client_manager()
    return await manager.get_tools()

def get_mcp_tools_sync() -> List[BaseTool]:
    """Get all MCP tools synchronously"""
    import asyncio
    try:
        # Try to get existing event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is running, we need to use asyncio.create_task
            # This is a workaround for synchronous calls in async context
            return asyncio.run_coroutine_threadsafe(get_mcp_tools(), loop).result()
        else:
            return loop.run_until_complete(get_mcp_tools())
    except RuntimeError:
        # No event loop, create a new one
        return asyncio.run(get_mcp_tools())

async def close_mcp_client():
    """Close MCP client connection"""
    global _mcp_client_manager
    if _mcp_client_manager:
        await _mcp_client_manager.close()
        _mcp_client_manager = None 