"""Score example action suffixes through the HTTP service."""
import argparse
import base64
import io
import json
import urllib.request

from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8011')
    parser.add_argument('--model', default='Qwen/Qwen3-VL-8B-Instruct')
    args = parser.parse_args()
    buffer = io.BytesIO()
    Image.new('RGB', (300, 300), 'red').save(buffer, format='PNG')
    request = dict(model=args.model, messages=[dict(role='user', content=[
        dict(type='image_url', image_url=dict(url='data:image/png;base64,' +
             base64.b64encode(buffer.getvalue()).decode())),
        dict(type='text', text='Describe the color of the image in one word.')])],
        candidates=[' red', ' blue', ' green'], candidate_batch_size=8)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(urllib.request.Request(
            args.url.rstrip('/') + '/v1/score_candidates',
            data=json.dumps(request).encode(), headers={'Content-Type': 'application/json'}),
            timeout=300) as response:
        result = json.load(response)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    assert result['best_index'] == 0, 'Expected red to have the highest score'


if __name__ == '__main__':
    main()
