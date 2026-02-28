import os
import uuid

import aiofiles
from aiohttp import web


# ---------- 原有上传接口 ----------
async def handle_upload(request):
    """
    处理文件上传的 POST 请求
    """
    reader = await request.multipart()
    field = await reader.next()
    if field:
        print(field.name, field.filename)

    if field and field.name == 'file' and field.filename:
        original_filename = field.filename
        # 安全处理：只保留文件名，防止路径遍历攻击
        original_filename = os.path.basename(original_filename)
        # 分离文件名和扩展名，保留原始后缀
        base, ext = os.path.splitext(original_filename)
        print(f'base={base}, ext={ext}')

        # 确保上传目录存在
        os.makedirs('./uploads', exist_ok=True)
        # 使用 UUID 作为新文件名
        new_filename = uuid.uuid4().hex + ext
        file_path = os.path.join('./uploads', new_filename)

        size = 0
        # 流式读取文件块并写入磁盘
        async with aiofiles.open(file_path, 'wb') as f:
            while True:
                chunk = await field.read_chunk()
                if not chunk:
                    break
                size += len(chunk)
                await f.write(chunk)
        print(f'file_path={file_path}, original_filename={original_filename}, size={size}')

        # 返回成功响应
        return web.json_response({
            'status': 'success',
            'filename': new_filename,
            'original_filename': original_filename,
            'size': size
        })
    else:
        return web.json_response({
            'status': 'error',
            'message': 'No file provided'
        }, status=400)


# ---------- 首页表单 ----------
async def handle_index(request):
    html = """
    <!DOCTYPE html>
    <html>
    <body>
        <h2>Upload a file</h2>
        <form action="/upload" method="post" enctype="multipart/form-data">
            <input type="file" name="file">
            <input type="submit" value="Upload">
        </form>
        <hr>
        <h2>Download from URL (use curl or Postman)</h2>
        <p>POST /download with JSON: {"file_url": "...", "file_ext": ".txt"}</p>
    </body>
    </html>
    """
    return web.Response(text=html, content_type='text/html')


# ---------- 创建应用并注册路由 ----------
app = web.Application()
app.router.add_get('/', handle_index)
app.router.add_post('/upload', handle_upload)

if __name__ == '__main__':
    web.run_app(app, host='127.0.0.1', port=8080)
