"""
兼容OpenAI的api接口服务
"""
"""
适配OpenAI API的兼容服务
"""
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from aiohttp import web, ClientSession
import aiohttp_cors
import uuid
import time


class OpenAICompatibleServer:
    def __init__(self, host: str = "localhost", port: int = 8000):
        self.host = host
        self.port = port
        self.app = web.Application()
        self.setup_routes()
        self.setup_cors()

        # 模拟的模型数据，使用OpenAI格式
        self.models = {
            "gpt-3.5-turbo": {
                "id": "gpt-3.5-turbo",
                "object": "model",
                "created": 1677610602,
                "owned_by": "openai",
                "permission": [
                    {
                        "id": "modelperm-123",
                        "object": "model_permission",
                        "created": 1677610602,
                        "allow_create_engine": False,
                        "allow_sampling": True,
                        "allow_logprobs": True,
                        "allow_search_indices": False,
                        "allow_view": True,
                        "allow_fine_tuning": False,
                        "organization": "*",
                        "group": None,
                        "is_blocking": False
                    }
                ],
                "root": "gpt-3.5-turbo",
                "parent": None
            },
            "text-embedding-ada-002": {
                "id": "text-embedding-ada-002",
                "object": "model",
                "created": 1671212500,
                "owned_by": "openai",
                "permission": [
                    {
                        "id": "modelperm-456",
                        "object": "model_permission",
                        "created": 1671212500,
                        "allow_create_engine": False,
                        "allow_sampling": True,
                        "allow_logprobs": True,
                        "allow_search_indices": False,
                        "allow_view": True,
                        "allow_fine_tuning": False,
                        "organization": "*",
                        "group": None,
                        "is_blocking": False
                    }
                ],
                "root": "text-embedding-ada-002",
                "parent": None
            },
            "gpt-4": {
                "id": "gpt-4",
                "object": "model",
                "created": 1687882411,
                "owned_by": "openai",
                "permission": [
                    {
                        "id": "modelperm-789",
                        "object": "model_permission",
                        "created": 1687882411,
                        "allow_create_engine": False,
                        "allow_sampling": True,
                        "allow_logprobs": True,
                        "allow_search_indices": False,
                        "allow_view": True,
                        "allow_fine_tuning": False,
                        "organization": "*",
                        "group": None,
                        "is_blocking": False
                    }
                ],
                "root": "gpt-4",
                "parent": None
            }
        }

    def setup_routes(self):
        """设置OpenAI API路由"""
        # 聊天完成接口
        self.app.router.add_post('/v1/chat/completions', self.handle_chat_completions)

        # 文本完成接口（旧版）
        self.app.router.add_post('/v1/completions', self.handle_completions)

        # 嵌入接口
        self.app.router.add_post('/v1/embeddings', self.handle_embeddings)

        # 模型列表接口
        self.app.router.add_get('/v1/models', self.handle_list_models)

        # 单个模型信息接口
        self.app.router.add_get('/v1/models/{model_id}', self.handle_retrieve_model)

        # 健康检查
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

    async def handle_chat_completions(self, request: web.Request) -> web.StreamResponse:
        """处理 /v1/chat/completions 请求 - OpenAI聊天完成接口"""
        try:
            data = await request.json()
            model = data.get('model', 'gpt-3.5-turbo')
            messages = data.get('messages', [])
            stream = data.get('stream', False)
            temperature = data.get('temperature', 0.7)
            max_tokens = data.get('max_tokens', 1000)

            # 获取最后一条用户消息
            user_messages = [m for m in messages if m['role'] == 'user']
            last_message = user_messages[-1]['content'] if user_messages else ""

            # 生成响应ID和时间戳
            response_id = f"chatcmpl-{uuid.uuid4().hex}"
            created_at = int(time.time())

            if stream:
                # 流式响应
                response = web.StreamResponse(
                    status=200,
                    reason='OK',
                    headers={
                        'Content-Type': 'text/event-stream',
                        'Cache-Control': 'no-cache',
                        'Connection': 'keep-alive',
                    }
                )
                await response.prepare(request)

                # 模拟流式聊天响应
                response_text = f"这是对你消息的回复：'{last_message[:50]}...'。这是一个模拟的OpenAI兼容响应。"
                sentences = response_text.split('。')

                for i, sentence in enumerate(sentences):
                    if sentence.strip():
                        # OpenAI流式响应格式
                        chunk_data = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": created_at,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "role": "assistant",
                                        "content": sentence + "。"
                                    },
                                    "finish_reason": None
                                }
                            ]
                        }
                        await response.write(f"data: {json.dumps(chunk_data)}\n\n".encode('utf-8'))
                        await asyncio.sleep(0.15)

                # 发送结束标志
                final_chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created_at,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }
                    ]
                }
                await response.write(f"data: {json.dumps(final_chunk)}\n\n".encode('utf-8'))
                await response.write(b"data: [DONE]\n\n")
                await response.write_eof()
                return response
            else:
                # 非流式响应
                return web.json_response({
                    "id": response_id,
                    "object": "chat.completion",
                    "created": created_at,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": f"这是对你消息的回复：'{last_message[:50]}...'。这是一个模拟的OpenAI兼容响应。"
                            },
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": {
                        "prompt_tokens": len(last_message.split()),
                        "completion_tokens": 50,
                        "total_tokens": len(last_message.split()) + 50
                    }
                })

        except Exception as e:
            return web.json_response({"error": {"message": str(e), "type": "server_error"}}, status=500)

    async def handle_completions(self, request: web.Request) -> web.StreamResponse:
        """处理 /v1/completions 请求 - OpenAI文本完成接口（旧版）"""
        try:
            data = await request.json()
            model = data.get('model', 'text-davinci-003')
            prompt = data.get('prompt', '')
            stream = data.get('stream', False)
            max_tokens = data.get('max_tokens', 100)
            temperature = data.get('temperature', 0.7)

            # 生成响应ID和时间戳
            response_id = f"cmpl-{uuid.uuid4().hex}"
            created_at = int(time.time())

            if stream:
                # 流式响应
                response = web.StreamResponse(
                    status=200,
                    reason='OK',
                    headers={
                        'Content-Type': 'text/event-stream',
                        'Cache-Control': 'no-cache',
                        'Connection': 'keep-alive',
                    }
                )
                await response.prepare(request)

                # 模拟流式生成
                words = prompt.split()
                for i, word in enumerate(words):
                    chunk_data = {
                        "id": response_id,
                        "object": "text_completion",
                        "created": created_at,
                        "model": model,
                        "choices": [
                            {
                                "text": word + " ",
                                "index": 0,
                                "logprobs": None,
                                "finish_reason": None
                            }
                        ]
                    }
                    await response.write(f"data: {json.dumps(chunk_data)}\n\n".encode('utf-8'))
                    await asyncio.sleep(0.1)

                # 发送结束标志
                final_chunk = {
                    "id": response_id,
                    "object": "text_completion",
                    "created": created_at,
                    "model": model,
                    "choices": [
                        {
                            "text": "",
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": "stop"
                        }
                    ]
                }
                await response.write(f"data: {json.dumps(final_chunk)}\n\n".encode('utf-8'))
                await response.write(b"data: [DONE]\n\n")
                await response.write_eof()
                return response
            else:
                # 非流式响应
                return web.json_response({
                    "id": response_id,
                    "object": "text_completion",
                    "created": created_at,
                    "model": model,
                    "choices": [
                        {
                            "text": f"基于你的提示 '{prompt[:50]}...' 生成的文本。这是一个模拟OpenAI响应。",
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": {
                        "prompt_tokens": len(prompt.split()),
                        "completion_tokens": 50,
                        "total_tokens": len(prompt.split()) + 50
                    }
                })

        except Exception as e:
            return web.json_response({"error": {"message": str(e), "type": "server_error"}}, status=500)

    async def handle_embeddings(self, request: web.Request) -> web.Response:
        """处理 /v1/embeddings 请求 - OpenAI嵌入向量接口"""
        try:
            data = await request.json()
            model = data.get('model', 'text-embedding-ada-002')
            input_text = data.get('input', '')

            # 确保input是列表形式
            if isinstance(input_text, str):
                inputs = [input_text]
            else:
                inputs = input_text

            # 生成模拟的嵌入向量（1536维，与text-embedding-ada-002相同）
            import random
            embeddings = []

            for text in inputs:
                random.seed(hash(text) % 10000)  # 基于输入生成确定性随机
                embedding = [random.gauss(0, 0.1) for _ in range(1536)]
                embeddings.append(embedding)

            return web.json_response({
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "embedding": embedding,
                        "index": i
                    }
                    for i, embedding in enumerate(embeddings)
                ],
                "model": model,
                "usage": {
                    "prompt_tokens": sum(len(text.split()) for text in inputs),
                    "total_tokens": sum(len(text.split()) for text in inputs)
                }
            })
        except Exception as e:
            return web.json_response({"error": {"message": str(e), "type": "server_error"}}, status=500)

    async def handle_list_models(self, request: web.Request) -> web.Response:
        """处理 /v1/models 请求 - 获取模型列表"""
        models_list = list(self.models.values())
        return web.json_response({
            "object": "list",
            "data": models_list
        })

    async def handle_retrieve_model(self, request: web.Request) -> web.Response:
        """处理 /v1/models/{model_id} 请求 - 获取单个模型信息"""
        model_id = request.match_info.get('model_id', '')

        if model_id in self.models:
            return web.json_response(self.models[model_id])
        else:
            return web.json_response({
                "error": {
                    "message": f"The model '{model_id}' does not exist",
                    "type": "invalid_request_error",
                    "param": "model_id",
                    "code": "model_not_found"
                }
            }, status=404)

    async def handle_health(self, request: web.Request) -> web.Response:
        """健康检查接口"""
        return web.json_response({
            "status": "healthy",
            "timestamp": datetime.now().isoformat()
        })

    def run(self):
        """启动服务器"""
        print(f"启动 OpenAI 兼容服务器在 http://{self.host}:{self.port}")
        print("可用接口:")
        print("  POST /v1/chat/completions   - 聊天完成（推荐）")
        print("  POST /v1/completions        - 文本完成（旧版）")
        print("  POST /v1/embeddings         - 生成嵌入向量")
        print("  GET  /v1/models             - 获取模型列表")
        print("  GET  /v1/models/{id}        - 获取特定模型信息")
        print("  GET  /health                - 健康检查")
        print("\n使用示例:")
        print("""
# 聊天完成（非流式）:
curl http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer sk-test" \\
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "你好，今天天气怎么样？"}
    ],
    "stream": false
  }'

# 聊天完成（流式）:
curl http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer sk-test" \\
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "你好，今天天气怎么样？"}
    ],
    "stream": true
  }'

# 获取模型列表:
curl http://localhost:8000/v1/models \\
  -H "Authorization: Bearer sk-test"

# 生成嵌入向量:
curl http://localhost:8000/v1/embeddings \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer sk-test" \\
  -d '{
    "model": "text-embedding-ada-002",
    "input": "这是一个测试文本"
  }'
        """)

        web.run_app(self.app, host=self.host, port=self.port)


# 使用示例
if __name__ == "__main__":
    # 创建服务器实例
    server = OpenAICompatibleServer(host="0.0.0.0", port=8000)

    # 启动服务器
    server.run()
