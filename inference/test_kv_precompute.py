import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from kv_reuse.precompute import PrecomputedKVEngine, PrecomputeError, extract_prepare_tokens
from test_kv_reuse import CharacterTokenizer


TEXT = (
    'Rules. ## The available action id list. action id 0: find. '
    'action id 162: drawer. ## Task Execution Example 0: first example. '
    '## Task Execution Example 1: second example. ## Guidelines follow. '
    '## Now the human instruction is: put the cup away.'
    ' You are supposed to output in json. The output json format should be {action: int}'
)


class PrecomputeProjectTest(unittest.TestCase):
    def test_extract_exact_tokens_without_special_tokens(self):
        tokens = extract_prepare_tokens(CharacterTokenizer(), TEXT, (1, 2, 4))
        self.assertEqual(''.join(map(chr, tokens[1][0].tolist())), 'Rules. ')
        actions = ''.join(map(chr, tokens[2][0].tolist()))
        self.assertTrue(actions.startswith('action id 0:'))
        self.assertNotIn('## The available action id list.', actions)
        self.assertIn('action id 0: find.', actions)
        self.assertNotIn('action id 162:', actions)
        example = ''.join(map(chr, tokens[4][0].tolist()))
        self.assertTrue(example.startswith('## Task Execution Example 0:'))
        self.assertIn('Example 1:', example)
        self.assertNotIn('Guidelines', example)

    def test_a2_precompute_ignores_scene_dependent_table_heading(self):
        first = TEXT.replace('## The available action id list.', '## The available action id (0 ~ 207) list.')
        second = TEXT.replace('## The available action id list.', '## The available action id (0 ~ 217) list.')
        tokenizer = CharacterTokenizer()
        self.assertTrue(torch.equal(
            extract_prepare_tokens(tokenizer, first, (2,))[2],
            extract_prepare_tokens(tokenizer, second, (2,))[2],
        ))

    def test_one_prepare_independent_forwards_and_no_missing_prepare_fallback(self):
        model = torch.nn.Linear(1, 1)
        model.model = SimpleNamespace(language_model=object())
        model.config = SimpleNamespace(text_config=SimpleNamespace(to_dict=lambda: {'num_hidden_layers': 2}))
        engine = PrecomputedKVEngine(model, CharacterTokenizer(), (1, 2, 4))
        context = dict(run_id='r', benchmark='eb-alf', eval_set='base', episode=1, call_id=1)
        with self.assertRaisesRegex(PrecomputeError, 'prepare_required'):
            engine.generate({}, [], context, {})
        seen = []
        def compute(language, ids):
            seen.append(ids.clone())
            return SimpleNamespace(keys=(torch.zeros(1, 2, ids.shape[1], 4),) * 2,
                                   values=(torch.ones(1, 2, ids.shape[1], 4),) * 2)
        with patch('kv_reuse.precompute.precompute_text_group', side_effect=compute):
            result = engine.prepare(TEXT, (1, 2, 4))
        self.assertEqual(len(seen), 3)
        self.assertEqual(set(result['groups']), {'A1', 'A2', 'A4'})
        with self.assertRaisesRegex(PrecomputeError, 'prepare_already_attempted'):
            engine.prepare(TEXT, (1, 2, 4))
        for episode in (1, 2):
            with self.assertRaisesRegex(PrecomputeError, 'precompute_id_mismatch'):
                engine.generate({}, [], {**context, 'episode': episode}, {})
        self.assertIs(engine.prepared, result)

    def test_wrapper_identifies_fatal_error(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/kv_reuse/run_embodiedbench.py'
        spec = importlib.util.spec_from_file_location('wrapper', path)
        wrapper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wrapper)
        self.assertTrue(wrapper.is_precompute_error(SimpleNamespace(
            body={'detail': {'code': 'kv_precompute_error', 'message': 'A1_token_mismatch'}})))
        self.assertFalse(wrapper.is_precompute_error(SimpleNamespace(body={'detail': 'invalid JSON'})))
        self.assertFalse(issubclass(wrapper.FatalPrecomputeError, Exception))


if __name__ == '__main__':
    unittest.main()
