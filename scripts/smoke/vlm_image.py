"""向本地 Transformers 服务发送合成图片，并打印完整响应。"""
import argparse
import base64
import io
import json
import os
import urllib.error
import urllib.request
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', default=os.getenv('VLM_BASE_URL', 'http://127.0.0.1:8001/v1'))
    parser.add_argument('--model-name', default=os.getenv('VLM_MODEL_NAME', 'Qwen/Qwen3-VL-8B-Instruct'))
    parser.add_argument('--timeout', type=int, default=240)
    args = parser.parse_args()

    buffer = io.BytesIO()
    Image.new('RGB', (224, 224), (255, 0, 0)).save(buffer, format='PNG')
    payload = dict(model=args.model_name, temperature=0, max_tokens=64,
                   messages=[dict(role='user', content=[
                       dict(type='text', text='What is the main color of this image?'),
                       dict(type='image_url', image_url=dict(url='data:image/png;base64,' +
                            base64.b64encode(buffer.getvalue()).decode()))])])
    endpoint = args.base_url.rstrip('/') + '/chat/completions'
    request = urllib.request.Request(endpoint, data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=args.timeout) as response:
            print(json.dumps(json.load(response), indent=2, ensure_ascii=False))
    except urllib.error.HTTPError as error:
        body = error.read().decode(errors='replace')
        raise RuntimeError(f'Model server returned HTTP {error.code}: {body}') from error


if __name__ == '__main__':
    main()
