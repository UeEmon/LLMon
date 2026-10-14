import argparse
import hashlib
import json
import platform
from pathlib import Path
from data_utils import validate_directory, encode_row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/cpu.json")
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    splits = validate_directory(args.data)
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError("出力先は既に使用されています。別のディレクトリを指定してください")
    out.mkdir(parents=True, exist_ok=True)
    import torch
    import transformers
    import peft
    from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TrainerCallback, set_seed
    from peft import LoraConfig, get_peft_model
    from datasets import Dataset
    torch.set_num_threads(cfg["threads"])
    set_seed(cfg["seed"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], trust_remote_code=False)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], torch_dtype=torch.float32, trust_remote_code=False,
        attn_implementation="eager")
    model.config.use_cache = False
    revision = getattr(model.config, "_commit_hash", None)
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"],
        lora_dropout=0.05, target_modules=["q_proj", "v_proj"], bias="none"))
    model.print_trainable_parameters()
    datasets = {k: Dataset.from_list([encode_row(row, tokenizer, cfg["max_length"]) for row in rows])
                for k, rows in splits.items()}

    def collate(features):
        length = max(len(f["input_ids"]) for f in features)
        values = {}
        for key, pad in (("input_ids", tokenizer.pad_token_id), ("attention_mask", 0), ("labels", -100)):
            values[key] = torch.tensor([f[key] + [pad] * (length - len(f[key])) for f in features])
        return values

    training_args = TrainingArguments(
        output_dir=str(out), use_cpu=True, fp16=False, bf16=False,
        max_steps=cfg["max_steps"], per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=1, gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"], optim="adamw_torch", report_to="none",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps", eval_steps=10, logging_steps=1, save_steps=10,
        save_total_limit=2, dataloader_num_workers=0, dataloader_pin_memory=False,
        seed=cfg["seed"])
    class WebProgress(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            record = {"step": state.global_step, "max_steps": state.max_steps,
                      **{k:v for k,v in (logs or {}).items() if isinstance(v,(int,float,str))}}
            with (out / "metrics.jsonl").open("a",encoding="utf-8") as handle:
                handle.write(json.dumps(record,ensure_ascii=False) + "\n")

    trainer = Trainer(model=model, args=training_args, train_dataset=datasets["train"],
                      eval_dataset=datasets["valid"], data_collator=collate, callbacks=[WebProgress()])
    manifest = {"config": cfg, "base_revision": revision, "engine": "pytorch-cpu",
                "platform": platform.platform(), "torch": torch.__version__,
                "transformers": transformers.__version__, "peft": peft.__version__,
                "data_sha256": {k: hashlib.sha256((Path(args.data) / f"{k}.jsonl").read_bytes()).hexdigest()
                                for k in splits}}
    (out / "run.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    trainer.train()
    trainer.save_model(str(out / "adapter"))
    tokenizer.save_pretrained(str(out / "adapter"))
    results = trainer.evaluate(datasets["test"], metric_key_prefix="test")
    (out / "test_metrics.json").write_text(json.dumps(results, indent=2))
    print(f"保存完了: {out}/adapter")


if __name__ == "__main__":
    main()
