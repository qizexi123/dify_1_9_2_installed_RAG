"""
兼容Ollama的api接口服务
"""
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
from aiohttp import web, ClientSession
import aiohttp_cors
import uuid


class OllamaCompatibleServer:
    def __init__(self, host: str = "localhost", port: int = 11434):
        self.host = host
        self.port = port
        self.app = web.Application()
        self.setup_routes()
        self.setup_cors()

        # 模拟的模型数据
        self.models = {
            "llama2": {
                "name": "llama2:latest",
                "modified_at": datetime.now().isoformat() + "Z",
                "size": 3820000000,
                "digest": "sha256:xxxxxxxx",
                "details": {
                    "format": "gguf",
                    "family": "llama",
                    "families": ["llama"],
                    "parameter_size": "7B",
                    "quantization_level": "Q4_0"
                }
            },
            "mistral": {
                "name": "mistral:latest",
                "modified_at": datetime.now().isoformat() + "Z",
                "size": 4100000000,
                "digest": "sha256:yyyyyyyy",
                "details": {
                    "format": "gguf",
                    "family": "mistral",
                    "families": ["mistral"],
                    "parameter_size": "7B",
                    "quantization_level": "Q4_K_M"
                }
            }
        }

    def setup_routes(self):
        """设置路由"""
        self.app.router.add_post('/api/generate', self.handle_generate)
        self.app.router.add_post('/api/chat', self.handle_chat)
        self.app.router.add_get('/api/tags', self.handle_tags)
        self.app.router.add_post('/api/embed', self.handle_embed)
        self.app.router.add_post('/api/pull', self.handle_pull)  # 可选：添加拉取模型接口
        self.app.router.add_post('/api/create', self.handle_create)  # 可选：添加创建模型接口

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

    async def handle_generate(self, request: web.Request) -> web.StreamResponse:
        """处理 /api/generate 请求"""
        try:
            data = await request.json()
            model = data.get('model', 'llama2:latest')
            prompt = data.get('prompt', '')
            stream = data.get('stream', False)
            options = data.get('options', {})

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
                    await response.write(
                        json.dumps({
                            "model": model,
                            "created_at": datetime.now().isoformat() + "Z",
                            "response": word + " ",
                            "done": False
                        }).encode('utf-8') + b'\n'
                    )
                    await asyncio.sleep(0.1)  # 模拟生成延迟

                # 发送结束标志
                await response.write(
                    json.dumps({
                        "model": model,
                        "created_at": datetime.now().isoformat() + "Z",
                        "response": "",
                        "done": True,
                        "context": [1, 2, 3],  # 模拟上下文
                        "total_duration": 5000000000,
                        "load_duration": 1000000000,
                        "prompt_eval_count": len(prompt.split()),
                        "prompt_eval_duration": 1000000000,
                        "eval_count": len(words),
                        "eval_duration": 4000000000
                    }).encode('utf-8') + b'\n'
                )
                await response.write_eof()
                return response
            else:
                # 非流式响应
                return web.json_response({
                    "model": model,
                    "created_at": datetime.now().isoformat() + "Z",
                    "response": f"基于你的提示 '{prompt[:50]}...' 生成的文本。这是一个模拟响应。",
                    "done": True,
                    "context": [1, 2, 3],
                    "total_duration": 5000000000,
                    "load_duration": 1000000000,
                    "prompt_eval_count": len(prompt.split()),
                    "prompt_eval_duration": 1000000000,
                    "eval_count": 100,
                    "eval_duration": 4000000000
                })

        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_chat(self, request: web.Request) -> web.StreamResponse:
        """处理 /api/chat 请求"""
        try:
            data = await request.json()
            model = data.get('model', 'llama2:latest')
            messages = data.get('messages', [])
            stream = data.get('stream', False)
            options = data.get('options', {})

            # 获取最后一条用户消息
            user_messages = [m for m in messages if m['role'] == 'user']
            last_message = user_messages[-1]['content'] if user_messages else ""

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
                response_text = f"这是对你消息的回复：'{last_message[:50]}...'。这是一个模拟的聊天响应。"
                sentences = response_text.split('。')

                for i, sentence in enumerate(sentences):
                    if sentence.strip():
                        await response.write(
                            json.dumps({
                                "model": model,
                                "created_at": datetime.now().isoformat() + "Z",
                                "message": {
                                    "role": "assistant",
                                    "content": sentence + "。"
                                },
                                "done": False
                            }).encode('utf-8') + b'\n'
                        )
                        await asyncio.sleep(0.15)

                # 发送结束标志
                await response.write(
                    json.dumps({
                        "model": model,
                        "created_at": datetime.now().isoformat() + "Z",
                        "message": {
                            "role": "assistant",
                            "content": ""
                        },
                        "done": True,
                        "total_duration": 6000000000,
                        "load_duration": 1000000000,
                        "prompt_eval_count": len(last_message.split()),
                        "prompt_eval_duration": 2000000000,
                        "eval_count": len(response_text.split()),
                        "eval_duration": 3000000000
                    }).encode('utf-8') + b'\n'
                )
                await response.write_eof()
                return response
            else:
                # 非流式响应
                return web.json_response({
                    "model": model,
                    "created_at": datetime.now().isoformat() + "Z",
                    "message": {
                        "role": "assistant",
                        "content": f"这是对你消息的回复：'{last_message[:50]}...'。这是一个模拟的聊天响应。"
                    },
                    "done": True,
                    "total_duration": 6000000000,
                    "load_duration": 1000000000,
                    "prompt_eval_count": len(last_message.split()),
                    "prompt_eval_duration": 2000000000,
                    "eval_count": 100,
                    "eval_duration": 3000000000
                })

        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_tags(self, request: web.Request) -> web.Response:
        """处理 /api/tags 请求 - 获取可用模型列表"""
        models_list = list(self.models.values())
        return web.json_response({
            "models": models_list
        })

    async def handle_embed(self, request: web.Request) -> web.Response:
        """处理 /api/embed 请求 - 生成嵌入向量"""
        try:
            data = await request.json()
            model = data.get('model', 'llama2:latest')
            input_text = data.get('input', '')

            # 生成模拟的嵌入向量（4096维，与常见模型相同）
            import random
            random.seed(hash(input_text) % 10000)  # 基于输入生成确定性随机

            # 模拟一个简单的嵌入向量
            embedding = [random.gauss(0, 0.1) for _ in range(4096)]

            return web.json_response({
                "model": model,
                "embeddings": [embedding] if isinstance(input_text, str) else [embedding] * len(input_text),
                "total_duration": 100000000,
                "load_duration": 50000000,
                "prompt_eval_count": len(input_text.split()) if isinstance(input_text, str) else sum(
                    len(t.split()) for t in input_text),
                "prompt_eval_duration": 50000000
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_pull(self, request: web.Request) -> web.StreamResponse:
        """处理 /api/pull 请求 - 拉取模型（模拟）"""
        data = await request.json()
        model = data.get('name', '')

        response = web.StreamResponse(
            status=200,
            reason='OK',
            headers={
                'Content-Type': 'application/json',
                'Cache-Control': 'no-cache',
            }
        )
        await response.prepare(request)

        # 模拟拉取过程
        steps = [
            {"status": "pulling manifest"},
            {"status": "downloading layers", "completed": 10, "total": 100},
            {"status": "downloading layers", "completed": 50, "total": 100},
            {"status": "downloading layers", "completed": 100, "total": 100},
            {"status": "verifying sha256 digest"},
            {"status": "writing manifest"},
            {"status": "success"}
        ]

        for step in steps:
            step.update({"status": f"pulling {model}: {step['status']}"})
            await response.write(json.dumps(step).encode('utf-8') + b'\n')
            await asyncio.sleep(0.5)

        await response.write_eof()
        return response

    async def handle_create(self, request: web.Request) -> web.StreamResponse:
        """处理 /api/create 请求 - 创建模型（模拟）"""
        data = await request.json()
        model = data.get('name', 'custom-model')

        response = web.StreamResponse(
            status=200,
            reason='OK',
            headers={
                'Content-Type': 'application/json',
                'Cache-Control': 'no-cache',
            }
        )
        await response.prepare(request)

        # 模拟创建过程
        steps = [
            {"status": "creating model"},
            {"status": "processing layers", "completed": 25, "total": 100},
            {"status": "processing layers", "completed": 75, "total": 100},
            {"status": "processing layers", "completed": 100, "total": 100},
            {"status": "writing model"},
            {"status": "success"}
        ]

        for step in steps:
            step.update({"status": f"creating {model}: {step['status']}"})
            await response.write(json.dumps(step).encode('utf-8') + b'\n')
            await asyncio.sleep(0.3)

        await response.write_eof()
        return response

    def run(self):
        """启动服务器"""
        print(f"启动 Ollama 兼容服务器在 http://{self.host}:{self.port}")
        print("可用接口:")
        print("  POST /api/generate    - 生成文本")
        print("  POST /api/chat        - 聊天")
        print("  GET  /api/tags        - 获取模型列表")
        print("  POST /api/embed       - 生成嵌入向量")
        print("  POST /api/pull        - 拉取模型")
        print("  POST /api/create      - 创建模型")

        web.run_app(self.app, host=self.host, port=self.port)


# 使用示例
if __name__ == "__main__":
    # 创建服务器实例
    server = OllamaCompatibleServer(host="0.0.0.0", port=11434)
    """
生成文本：
# 非流式
curl http://localhost:11434/api/generate -d '{
  "model": "llama2:latest",
  "prompt": "你好，今天天气怎么样？",
  "stream": false
}'
# 流式
curl http://localhost:11434/api/generate -d '{
  "model": "llama2:latest",
  "prompt": "你好，今天天气怎么样？",
  "stream": true
}'

聊天：
curl http://localhost:11434/api/chat -d '{
  "model": "llama2:latest",
  "messages": [
    {"role": "user", "content": "你好"}
  ],
  "stream": false
}'

获取模型列表：
curl http://localhost:11434/api/tags

生成嵌入向量：
curl http://localhost:11434/api/embed -d '{
  "model": "llama2:latest",
  "input": "这是一个测试文本"
}'
    """
    # 启动服务器
    server.run()
