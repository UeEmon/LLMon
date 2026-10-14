import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--base-only", action="store_true")
    args = parser.parse_args()
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    run = Path(args.run)
    manifest = json.loads((run / "run.json").read_text())
    torch.set_num_threads(manifest["config"]["threads"])
    tokenizer = AutoTokenizer.from_pretrained(run / "adapter")
    model = AutoModelForCausalLM.from_pretrained(manifest["config"]["model"],
        revision=manifest.get("base_revision"), torch_dtype=torch.float32, trust_remote_code=False)
    if not args.base_only:
        model = PeftModel.from_pretrained(model, run / "adapter")
    model.eval()
    prompt = tokenizer.apply_chat_template([
        {"role": "system", "content": "日本語で簡潔に回答してください。"},
        {"role": "user", "content": args.prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.inference_mode():
        tokens = model.generate(**inputs, max_new_tokens=128, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
    print(tokenizer.decode(tokens[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
