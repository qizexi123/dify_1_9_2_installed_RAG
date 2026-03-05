"""
兼容OpenAI的api接口服务（异步转发到真实LLM后端）
"""
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional

# pip install aiologger
from aiologger import Logger
from aiologger.handlers.files import AsyncFileHandler

# 不可以重定向到普通文件中
# logging.basicConfig(level=logging.DEBUG)
# logger = Logger.with_default_handlers(level=logging.DEBUG)


# 添加文件处理器
# 不再使用 with_default_handlers，而是自定义处理器
logger = Logger(name='openai_compatible')
# 注意：使用 AsyncFileHandler 时，日志写入实际上是线程池操作，虽然不是真正的非阻塞，但不会阻塞事件循环太长时间，对于一般应用足够。
file_handler = AsyncFileHandler(filename='openai_compatible_api_main.log')
logger.add_handler(file_handler)

from aiohttp import web, ClientSession
import aiohttp_cors


class OpenAICompatibleServer:
    def __init__(self, host: str = "localhost", port: int = 8000,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        self.host = host
        self.port = port
        self.app = web.Application(logger=logger)
        self.setup_routes()
        self.setup_cors()

        # 从环境变量或参数获取目标API配置
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.ofox.ai/v1")

        # 如果没有设置API密钥，打印警告
        if not self.api_key:
            logger.warning("警告: 未设置 OPENAI_API_KEY，转发请求将失败")

        # 会话管理（应用启动时创建，关闭时清理）
        self.session: Optional[ClientSession] = None
        self.app.on_startup.append(self.on_startup)
        self.app.on_cleanup.append(self.on_cleanup)

    async def on_startup(self, app):
        """应用启动时创建HTTP会话"""
        self.session = ClientSession()

    async def on_cleanup(self, app):
        """应用关闭时关闭HTTP会话"""
        if self.session:
            await self.session.close()
        # 重要：应用退出前关闭logger，确保所有日志都被刷新
        await logger.shutdown()

    def setup_routes(self):
        """设置OpenAI API路由"""
        self.app.router.add_post('/v1/chat/completions', self.handle_chat_completions)
        self.app.router.add_post('/v1/completions', self.handle_completions)
        self.app.router.add_post('/v1/embeddings', self.handle_embeddings)
        self.app.router.add_get('/v1/models', self.handle_list_models)
        self.app.router.add_get('/v1/models/{model_id}', self.handle_retrieve_model)
        self.app.router.add_get('/health', self.handle_health)

    def setup_cors(self):
        """设置CORS，允许跨域请求"""
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )
        })
        for route in list(self.app.router.routes()):
            cors.add(route)

    async def _forward_request(self, path: str, data: Dict[str, Any],
                               client_request: web.Request) -> web.StreamResponse:
        """
        通用请求转发核心逻辑
        :param path: 目标API路径（如 '/chat/completions'）
        :param data: 要转发的请求体
        :param client_request: 原始客户端请求（用于获取stream参数）
        :return: 转发后的响应（可能是流式或非流式）
        """
        stream = data.get('stream', False)
        url = self.base_url.rstrip('/') + path
        await logger.debug(f"\n\n")
        await logger.debug(f"-" * 50)
        await logger.debug(f"body: {data}")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            async with self.session.post(url, json=data, headers=headers) as resp:
                if stream:
                    # 流式响应：逐行转发SSE数据
                    response = web.StreamResponse(
                        status=resp.status,
                        reason=resp.reason,
                        headers={
                            'Content-Type': 'text/event-stream',
                            'Cache-Control': 'no-cache',
                            'Connection': 'keep-alive',
                        }
                    )
                    await response.prepare(client_request)

                    # 逐行读取外部API的流式响应并转发
                    async for line in resp.content:
                        if line:  # 过滤空行
                            await response.write(line)
                            await logger.debug(f"line: {line}")
                    await response.write_eof()
                    return response
                else:
                    # 非流式响应：读取完整JSON并返回
                    result = await resp.json()
                    await logger.debug(f"result: {result}")
                    return web.json_response(result, status=resp.status)

        except Exception as e:
            # 转发失败时返回OpenAI风格错误
            return web.json_response({
                "error": {
                    "message": f"Forwarding error: {str(e)}",
                    "type": "proxy_error"
                }
            }, status=502)

    async def handle_chat_completions(self, request: web.Request) -> web.StreamResponse:
        """处理 /v1/chat/completions - 转发到真实LLM"""
        try:
            data = await request.json()
            return await self._forward_request('/chat/completions', data, request)
        except Exception as e:
            return web.json_response({
                "error": {"message": str(e), "type": "server_error"}
            }, status=500)

    async def handle_completions(self, request: web.Request) -> web.StreamResponse:
        """处理 /v1/completions - 转发到真实LLM（旧版文本补全）"""
        try:
            data = await request.json()
            return await self._forward_request('/completions', data, request)
        except Exception as e:
            return web.json_response({
                "error": {"message": str(e), "type": "server_error"}
            }, status=500)

    async def handle_embeddings(self, request: web.Request) -> web.Response:
        """处理 /v1/embeddings - 转发到真实嵌入服务"""
        try:
            data = await request.json()
            url = self.base_url.rstrip('/') + '/embeddings'
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            async with self.session.post(url, json=data, headers=headers) as resp:
                result = await resp.json()
                return web.json_response(result, status=resp.status)
        except Exception as e:
            return web.json_response({
                "error": {"message": str(e), "type": "server_error"}
            }, status=500)

    async def handle_list_models(self, request: web.Request) -> web.Response:
        """处理 /v1/models - 从目标API获取模型列表"""
        url = self.base_url.rstrip('/') + '/models'
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            async with self.session.get(url, headers=headers) as resp:
                result = await resp.json()
                return web.json_response(result, status=resp.status)
        except Exception as e:
            return web.json_response({
                "error": {"message": str(e), "type": "proxy_error"}
            }, status=502)

    async def handle_retrieve_model(self, request: web.Request) -> web.Response:
        """处理 /v1/models/{model_id} - 获取单个模型信息"""
        model_id = request.match_info.get('model_id', '')
        url = f"{self.base_url.rstrip('/')}/models/{model_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            async with self.session.get(url, headers=headers) as resp:
                result = await resp.json()
                return web.json_response(result, status=resp.status)
        except Exception as e:
            return web.json_response({
                "error": {"message": str(e), "type": "proxy_error"}
            }, status=502)

    async def handle_health(self, request: web.Request) -> web.Response:
        """健康检查接口"""
        return web.json_response({
            "status": "healthy",
            "timestamp": datetime.now().isoformat()
        })

    def run(self):
        """启动服务器"""
        print(f"启动 OpenAI 代理服务器在 http://{self.host}:{self.port}")
        print("将转发请求到:", self.base_url)
        print("可用接口:")
        print("  POST /v1/chat/completions   - 聊天完成（流式/非流式）")
        print("  POST /v1/completions        - 文本完成（旧版）")
        print("  POST /v1/embeddings         - 生成嵌入向量")
        print("  GET  /v1/models             - 获取模型列表")
        print("  GET  /v1/models/{id}        - 获取特定模型信息")
        print("  GET  /health                - 健康检查")
        print("\n使用示例:")
        print("""
# 聊天完成（流式）:
curl http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer sk-test" \\
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "你好"}],
    "stream": true
  }'
        """)
        web.run_app(self.app, host=self.host, port=self.port)


if __name__ == "__main__":
    # ollama
    base_url = "http://127.0.0.1:11434/v1"

    api_key = "..."
    server = OpenAICompatibleServer(host="0.0.0.0", port=8000, api_key=api_key, base_url=base_url)
    server.run()
