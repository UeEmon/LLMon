"""Shared data validation; dependency-free so it also runs on macOS."""
import json
from pathlib import Path


def read_rows(path):
    rows = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            messages = row["messages"]
            roles = [m["role"] for m in messages]
            if roles not in (["user", "assistant"], ["system", "user", "assistant"]):
                raise ValueError("system(任意)→user→assistantの1往復形式が必要です")
            if any(not isinstance(m["content"], str) or not m["content"].strip() for m in messages):
                raise ValueError("contentは空でない文字列が必要です")
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"{path}:{line_no}: {exc}") from exc
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: データが空です")
    return rows


def validate_directory(directory):
    splits = {name: read_rows(Path(directory) / f"{name}.jsonl") for name in ("train", "valid", "test")}
    seen = {}
    for name, rows in splits.items():
        for row in rows:
            # Split leakage is checked by prompt, even when answers differ.
            key = json.dumps(row["messages"][:-1], ensure_ascii=False, sort_keys=True)
            if key in seen:
                raise ValueError(f"質問の重複: {seen[key]} / {name}")
            seen[key] = name
    return splits


def encode_row(row, tokenizer, max_length):
    prompt = tokenizer.apply_chat_template(row["messages"][:-1], tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(row["messages"][-1]["content"], add_special_tokens=False)["input_ids"]
    if len(prompt_ids) + len(answer_ids) + 1 > max_length:
        raise ValueError("サンプルがmax_lengthを超えます。文章を短くするか設定を増やしてください")
    if not answer_ids:
        raise ValueError("回答トークンがありません")
    ids = prompt_ids + answer_ids + [tokenizer.eos_token_id]
    return {"input_ids": ids, "attention_mask": [1] * len(ids),
            "labels": [-100] * len(prompt_ids) + answer_ids + [tokenizer.eos_token_id]}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="?", default="data")
    args = parser.parse_args()
    print({k: len(v) for k, v in validate_directory(args.directory).items()})
