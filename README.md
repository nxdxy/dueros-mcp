# 小度智能终端 MCP Server 

[![MCP](https://img.shields.io/badge/MCP-Compatible-blue)](https://modelcontextprotocol.io/)

一个基于 Model Context Protocol (MCP) 的小度智能终端服务，让AI应用能够无缝控制小度设备。支持Claude、Cursor、Cline等所有兼容MCP协议的平台。

## ✨ 特性

- **开放式控制** - 通过自然语言指令控制小度设备
- **语音播报** - 让小度设备朗读指定文本
- **实时拍照** - 获取支持摄像头的小度设备的实时图像
- **设备管理** - 获取用户绑定的在线设备列表
- **资源推送** - 推送图片、视频、音频到小度设备

## 📋 目录

- [快速开始](#-快速开始)
- [接入方式](#-接入方式)
- [工具说明](#-工具说明)

## 🚀 快速开始

### 前提条件

- 百度开发者账号
- 小度智能设备
- 支持MCP的AI工具（Claude、Cursor、Cline等）

### 获取访问令牌

1. 访问[百度开发者平台](https://developer.dueros.baidu.com/)
2. 按照[接入授权文档](https://developer.dueros.baidu.com/doc/dueros-bot-platform//mcp-server/prepare/auth-intro_markdown)获取`access_token`
3. 保存好您的`access_token`，后续配置时需要使用

## 🔌 接入方式

我们提供两种接入方式以满足不同场景的需求：

### 方式一：StreamableHTTP 接入

StreamableHTTP模式直接连接到我们的服务器（要求你使用的客户端支持StreamableHTTP方式， 并支持传递headers参数）。

#### 支持的工具
- **Cursor, Cherry Studio** ✅（推荐）
- **自定义客户端** ✅

#### 配置步骤
**以 Cursor 为例：**

1. 打开 Cursor → 设置 → MCP & Integrations
2. 在Mcp Tools配置项中添加：
```json
{
  "mcpServers": {
    "xiaodu-mcp": {
      "url": "https://xiaodu.baidu.com/dueros_mcp_server/mcp/",
      "headers": {
        "ACCESS_TOKEN": "${your_access_token}"
      }
    }
  }
}
```
3. 在会话中选择 Agent 模式开始使用

**以 Cherry Studio 为例：**

1. 打开 Cherry Studio → 设置 → MCP服务器
2. 添加新的MCP服务器：
   - **名称**: `xiaodu_mcp`
   - **类型**: `可流式传输的HTTP（streamableHttp）`
   - **URL**: `https://xiaodu.baidu.com/dueros_mcp_server/mcp/`
   - **请求头**: `ACCESS_TOKEN=${your_access_token}`

3. 启用服务器并开始使用

**开发者自定义Agent:**
1. 参考本代码库提供的[Demo](clients)
2. Demo中共提供了两种封装程度的使用示例
    - 直接调用mcp-server：[simple_chatbot](clients/simple_chatbot) 使用官方mcp python库直接调用mcp-server
    - Agent调用mcp-server : [art_gallery_agent](clients/art_gallery_agent) 使用langGraph框架在Agent应用中调用mcp-server

### 方式二：Stdio 接入

Stdio模式需要通过本地代理服务连接，这里推荐使用mcp-proxy。

#### 支持的工具
- **Claude Desktop** ✅
- **Cline** ✅
- **其他支持MCP的工具** ✅

#### 配置步骤

**第一步：安装代理服务**

使用开源项目 [mcp-proxy](https://github.com/sparfenyuk/mcp-proxy)：

```bash
# 方式1：使用uv安装（推荐）
uv tool install mcp-proxy

# 方式2：使用pipx安装
pipx install mcp-proxy

# 查看安装路径
which mcp-proxy
```

**第二步：配置AI工具**

**Claude Desktop 配置示例：**

编辑 `claude_desktop_config.json`：

```json
{
  "mcpServers": {
    "xiaodu_mcp": {
      "command": "/path/to/mcp-proxy",
      "args": [
        "https://xiaodu.baidu.com/dueros_mcp_server/mcp/",
        "--headers",
        "ACCESS_TOKEN",
        "your_access_token_here",
        "--transport",
        "streamablehttp"
      ]
    }
  }
}
```

## 🛠️ 工具说明

本MCP服务器提供以下4个工具，让您能够完全控制小度智能设备：

### 1. 获取设备列表 (`list_user_devices`)

获取与已验证用户关联且当前在线的设备列表。

#### 参数
- 无需参数

#### 返回值
- `List[Dict[str, Any]]`: 设备信息列表

---

### 2. 设备控制 (`control_xiaodu`)

向小度设备发送语音指令，设备将像听到用户说话一样执行该指令。

#### 参数
- `command` (string, required): 要发送给设备的语音指令文本
- `cuid` (string, required): 设备标识符
- `client_id` (string, required): 客户端标识符

#### 返回值
- `string`: 小度设备的响应或执行结果

---

### 3. 语音播报 (`xiaodu_speak`)

让小度设备朗读指定的文本内容。

#### 参数
- `text` (string, required): 要朗读的文本内容
- `cuid` (string, required): 设备标识符
- `client_id` (string, required): 客户端标识符

#### 返回值
- `string`: 操作执行状态

---

### 4. 设备拍照 (`xiaodu_take_photo`)

触发支持摄像头的小度设备进行拍照并返回图像内容。

#### 参数
- `cuid` (string, required): 设备标识符
- `client_id` (string, required): 客户端标识符

#### 返回值
- `ImageContent`: 图像内容对象
  - `content` (string): Base64编码的JPEG格式图像数据
  - `content_type` (string): 图像内容类型，固定为 "image/jpeg"

### 5. 资源推送 (`push_resource_to_xiaodu`)

推图片/图片+背景音/视频/音频 到小度设备。

#### 参数
- `resource_type` (string, required): 资源类型，支持 "image"、"image_with_bgm"、"video"、"audio"
- `cuid` (string, required): 设备CUID
- `client_id` (string, required): 设备client_id
- `image_url` (string, required): 图片地址（image / image_with_bgm 必填）
- `bgm_url` (string, required): 背景音地址（image_with_bgm 必填）
- `video_url` (string, required): 视频地址（video 必填）
- `audio_url` (string, required): 音频地址（audio 必填）
- `timeout` (int, optional): 超时时间（秒）

## 客户端示例

Simple Chatbot: [clients/simple_chatbot/README.md](clients/simple_chatbot/README.md)

Art Gallery Agent: [clients/art_gallery_agent/README.md](clients/art_gallery_agent/README.md)

Xiaodu Assistant Agent: [clients/assistant_agent/README.md](clients/assistant_agent/README.md)

## 🙏 致谢

- [Model Context Protocol](https://modelcontextprotocol.io/) - MCP协议标准
- [mcp-proxy](https://github.com/sparfenyuk/mcp-proxy) - 优秀的MCP代理工具
