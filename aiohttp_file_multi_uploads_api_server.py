"""
支持多文件上传版本
"""
import os
import uuid
import aiofiles
from aiohttp import web


# ---------- 批量上传接口 ----------
async def handle_upload(request):
    """
    处理批量文件上传的 POST 请求
    支持 multipart/form-data 中的多个文件字段
    返回每个文件的保存信息
    """
    reader = await request.multipart()
    uploaded_files = []  # 存储所有成功上传的文件信息

    print()
    print("=" * 50)
    while True:
        field = await reader.next()
        if field is None:
            break  # 所有字段处理完毕

        # 只处理文件字段（包含 filename 的字段）
        if field.filename:
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
            # 流式读取文件块并异步写入磁盘
            async with aiofiles.open(file_path, 'wb') as f:
                while True:
                    chunk = await field.read_chunk()  # 默认 8192 字节
                    if not chunk:
                        break
                    size += len(chunk)
                    await f.write(chunk)
            print(f'size={size}, file_path={file_path}')

            # 记录文件信息
            uploaded_files.append({
                'filename': new_filename,
                'original_filename': original_filename,
                'size': size
            })

            print(f'Saved: {original_filename} -> {new_filename} ({size} bytes)')
            print("-" * 50)

        else:
            # 忽略非文件字段（如普通文本字段）
            # 如果需要读取内容，可以在这里处理，但通常批量上传只关心文件
            pass

    if not uploaded_files:
        return web.json_response({
            'status': 'error',
            'message': 'No files provided'
        }, status=400)

    return web.json_response({
        'status': 'success',
        'file_count': len(uploaded_files),
        'total_size': sum([v['size'] for v in uploaded_files]),
        'files': uploaded_files
    })


# ---------- 首页表单（演示批量上传）----------
async def handle_index(request):
    html = """
    <!DOCTYPE html>
    <html>
    <body>
        <h2>Upload multiple files</h2>
        <form action="/upload" method="post" enctype="multipart/form-data">
            <input type="file" name="files" multiple>
            <input type="submit" value="Upload">
        </form>
        <hr>
        <p>Use curl to test batch upload:</p>
        <pre>
curl -X POST http://127.0.0.1:8080/upload \\
     -F "file1=@/path/to/file1.txt" \\
     -F "file2=@/path/to/file2.jpg"
        </pre>
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
