import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from data_utils import encode_row, validate_directory


class FakeTokenizer:
    eos_token_id = 2
    def apply_chat_template(self, messages, **kwargs):
        return "prompt"
    def __call__(self, text, **kwargs):
        return {"input_ids": [10, 11, 12] if text == "prompt" else [20, 21]}


class DataTests(unittest.TestCase):
    def test_masks_prompt_but_trains_answer_and_eos(self):
        row = {"messages": [{"role": "user", "content": "質問"}, {"role": "assistant", "content": "回答"}]}
        encoded = encode_row(row, FakeTokenizer(), 20)
        self.assertEqual(encoded["labels"], [-100, -100, -100, 20, 21, 2])
        self.assertEqual(len(encoded["input_ids"]), len(encoded["labels"]))

    def test_rejects_truncation_of_answer(self):
        row = {"messages": [{"role": "user", "content": "質問"}, {"role": "assistant", "content": "回答"}]}
        with self.assertRaises(ValueError):
            encode_row(row, FakeTokenizer(), 5)

    def test_detects_prompt_leakage_even_with_different_answers(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, answer in [("train", "A"), ("valid", "B"), ("test", "C")]:
                row = {"messages": [{"role": "user", "content": "同じ質問"}, {"role": "assistant", "content": answer}]}
                Path(tmp, name + ".jsonl").write_text(json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                validate_directory(tmp)

    def test_sample_splits_are_independent(self):
        splits = validate_directory(Path(__file__).resolve().parents[1] / "data")
        self.assertEqual({k: len(v) for k, v in splits.items()}, {"train": 24, "valid": 4, "test": 4})


if __name__ == "__main__":
    unittest.main()
