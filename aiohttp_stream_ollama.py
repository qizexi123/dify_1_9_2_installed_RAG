import sys

import aiohttp
import asyncio
import json
from typing import AsyncGenerator


async def stream_ollama(
        model: str = "gpt-oss:20b",
        prompt: str = "解释量子力学中的叠加原理。",
        stream_url: str = "http://192.168.3.250:11434/api/generate"
) -> AsyncGenerator[str, None]:
    """
    异步流式调用 Ollama 的 /api/generate 接口，逐块 yield 响应内容
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,  # 关键！启用流式
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(stream_url, json=payload) as resp:
                # 检查 HTTP 状态
                if resp.status != 200:
                    error_text = await resp.text()
                    raise aiohttp.ClientError(f"Ollama API 返回错误状态 {resp.status}: {error_text}")

                # 使用 aiohttp 的异步行读取器
                async for line in resp.content:
                    # 跳过空行（SSE 常见）
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        # 解析 JSON 行
                        data = json.loads(line.decode("utf-8"))

                        # Ollama 每次返回的字段：{'model':..., 'created_at':..., 'response': ..., 'done': ...}
                        response_text = data.get("response", "")

                        # 如果是结束标志，可选择返回完成信息或静默忽略（看需求）
                        if data.get("done", False):
                            # 例如可额外 yield 总结信息
                            if "total_duration" in data:
                                print(
                                    f"\n[模型流式完成] 耗时: {data['total_duration'] / 1e9:.2f}s, token 数: {data.get('eval_count', '?')}")
                            break

                        # 只 yield 生成的内容（非结束标记）
                        if response_text:
                            yield response_text

                    except json.JSONDecodeError as e:
                        print(f"[警告] 无法解析流数据行: {line!r} | 错误: {e}", file=sys.stderr)
                        continue

        except aiohttp.ClientError as e:
            print(f"请求失败: {e}", file=sys.stderr)
        except Exception as e:
            print(f"未预期错误: {e}", file=sys.stderr)


# 🔧 使用示例
async def main():
    print("正在请求 Ollama (LLaMA3)...", end="\n\n")

    async for chunk in stream_ollama(
            model="gpt-oss:20b",
            prompt="用 3句话解释牛顿第一定律："
    ):
        # 实时打印（不换行，像打字机效果）
        print(chunk, end="", flush=True)

    print("\n✅ 流式生成完毕！")


if __name__ == "__main__":
    asyncio.run(main())
