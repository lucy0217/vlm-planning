"""向本地 Transformers 服务发送合成图片，并打印完整响应。"""
import base64
import io
import json
import urllib.request
from PIL import Image

buffer = io.BytesIO()
Image.new('RGB', (224, 224), (255, 0, 0)).save(buffer, format='PNG')
payload = dict(model='Qwen/Qwen3-VL-8B-Instruct', temperature=0, max_tokens=64,
               messages=[dict(role='user', content=[
                   dict(type='text', text='What is the main color of this image?'),
                   dict(type='image_url', image_url=dict(url='data:image/png;base64,' +
                        base64.b64encode(buffer.getvalue()).decode()))])])
request = urllib.request.Request('http://127.0.0.1:8001/v1/chat/completions',
                                 data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(request, timeout=240) as response:
    print(json.dumps(json.load(response), indent=2, ensure_ascii=False))
