#可以删除，检查的代码
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import torch

from kv_reuse.cache_manager import EpisodeCacheManager, EpisodeKey
from kv_reuse.metrics import JSONLMetricsWriter
from kv_reuse.segments import locate_prompt_segments


class Encoding:
    def __init__(self, text):
        self.input_ids = [ord(character) for character in text]
        self.offset_mapping = [(index, index + 1) for index in range(len(text))]


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        return Encoding(text)


class KVReuseProjectTest(unittest.TestCase):
    def test_embodiedbench_prompt_segments(self):
        text = (
            "System rules. ## The available action id list. action id 0: find. "
            "action id 162: dynamic. ## Task Execution Example 0: example. "
            "## Guidelines follow. ## Now the human instruction is: put the cup away."
            " You are supposed to output in json. Dynamic first call. "
            "The output json format should be {action: int}"
        )
        full_ids = torch.tensor([[999999, *[ord(character) for character in text], 999998]])
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        located = locate_prompt_segments(CharacterTokenizer(), full_ids, messages)
        self.assertEqual(located.model.a_start, 1)
        self.assertEqual(located.model.c_end, full_ids.shape[-1] - 1)
        self.assertGreater(located.token_counts["B"], 0)
        self.assertEqual(located.token_counts["S"], 1)
        self.assertEqual(sum(located.token_counts[f"A{i}"] for i in range(1, 7)), located.token_counts["A"])
        for parts in ((1, 5), (1, 4, 5), (1, 2, 5), (1, 2, 3, 4, 5, 6)):
            mask = located.reuse_mask(parts)
            self.assertEqual(
                int(mask.sum()), sum(located.token_counts[f"A{i}"] for i in parts)
            )

    def test_episode_change_releases_canonical_state(self):
        manager = EpisodeCacheManager()
        first = EpisodeKey("run", "eb-alf", "base", "1")
        second = EpisodeKey("run", "eb-alf", "base", "2")
        manager.set_first(first, "1", "canonical")
        self.assertEqual(manager.state_for(first), "canonical")
        self.assertIsNone(manager.state_for(second))
        self.assertIsNone(manager.entry)

    def test_metrics_are_routed_to_the_requested_run_directory(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "runs"
            run = root / "experiment"
            writer = JSONLMetricsWriter(None, runs_root=root)
            writer.write({"cache_status": "hit"}, context={"run_dir": str(run)})
            self.assertTrue((run / "server_metrics.jsonl").is_file())

            with self.assertRaisesRegex(ValueError, "run_dir must be inside"):
                writer.resolve_path({"run_dir": str(Path(temporary) / "outside")})


if __name__ == "__main__":
    unittest.main()
