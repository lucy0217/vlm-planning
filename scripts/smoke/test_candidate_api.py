"""Integration tests against a running scoring service on port 8011."""
import base64
import copy
import io
import json
import os
import unittest
import urllib.error
import urllib.request

from PIL import Image


class CandidateAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.url = os.getenv('VLM_SCORE_URL', 'http://127.0.0.1:8011').rstrip('/')
        cls.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        buffer = io.BytesIO()
        Image.new('RGB', (300, 300), 'red').save(buffer, format='PNG')
        cls.payload = dict(model='Qwen/Qwen3-VL-8B-Instruct', candidates=[' red', ' blue'],
            messages=[dict(role='user', content=[
                dict(type='image_url', image_url=dict(url='data:image/png;base64,' +
                    base64.b64encode(buffer.getvalue()).decode())),
                dict(type='text', text='Describe the color of the image.')])])

    def request(self, payload):
        request = urllib.request.Request(self.url + '/v1/score_candidates',
            data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        with self.opener.open(request, timeout=300) as response:
            return json.load(response)

    def test_invalid_requests(self):
        cases = [dict(candidates=[]), dict(candidates=['']), dict(candidate_batch_size=0),
                 dict(score_function='invalid'), dict(model='wrong-model'), dict(messages=[])]
        for changes in cases:
            with self.subTest(changes=changes):
                payload = dict(self.payload, **changes)
                with self.assertRaises(urllib.error.HTTPError) as error:
                    self.request(payload)
                self.assertEqual(error.exception.code, 404 if 'model' in changes else 400)

    def test_context_limit(self):
        payload = dict(self.payload, candidates=[' word' * 17000])
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request(payload)
        self.assertEqual(error.exception.code, 400)

    def test_ties_and_subsequent_valid_request(self):
        payload = dict(self.payload, candidates=[' red', ' red', ' blue'])
        result = self.request(payload)
        self.assertEqual(result['best_index'], 0)
        self.assertEqual(result['scores'][0]['score'], result['scores'][1]['score'])
        self.assertEqual(self.request(self.payload)['best_index'], 0)

    def test_fixed_size_new_frame(self):
        payload = copy.deepcopy(self.payload)
        buffer = io.BytesIO()
        Image.new('RGB', (300, 300), 'blue').save(buffer, format='PNG')
        payload['messages'][0]['content'][0]['image_url']['url'] = (
            'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode())
        self.assertEqual(self.request(payload)['best_index'], 1)
        self.assertEqual(self.request(self.payload)['best_index'], 0)


if __name__ == '__main__':
    unittest.main()
