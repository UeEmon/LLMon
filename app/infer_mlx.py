import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True)
parser.add_argument("--prompt", required=True)
parser.add_argument("--base-only", action="store_true")
args = parser.parse_args()
from mlx_lm import load, generate
run = Path(args.run)
model, tokenizer = load((run / "base-model.txt").read_text().strip(),
                        adapter_path=None if args.base_only else str(run / "adapter"))
prompt = tokenizer.apply_chat_template([
    {"role": "system", "content": "日本語で簡潔に回答してください。"},
    {"role": "user", "content": args.prompt}], tokenize=False,
    add_generation_prompt=True, enable_thinking=False)
print(generate(model, tokenizer, prompt=prompt, max_tokens=128, verbose=False))
