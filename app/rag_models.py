"""Lazy CPU-only models. No external document API and no remote custom code."""
import json
from pathlib import Path
import threading


class Embedder:
    def __init__(self, name, threads=6):
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(threads)
        self.model = SentenceTransformer(name, device='cpu', trust_remote_code=False)
        revision = getattr(self.model[0].auto_model.config, '_commit_hash', None) or 'local'
        self.identity = f'{name}@{revision}'
        self.lock = threading.Lock()

    def token_count(self, text):
        return len(self.model.tokenizer('passage: ' + text, truncation=False)['input_ids'])

    def passages(self, texts):
        if any(self.token_count(t) > 512 for t in texts):
            raise ValueError('埋め込みの512トークン上限を超えています。チャンク長を小さくしてください')
        with self.lock:
            return self.model.encode(['passage: ' + t for t in texts], batch_size=16,
                                     normalize_embeddings=True, show_progress_bar=False)

    def query(self, text):
        if len(self.model.tokenizer('query: ' + text, truncation=False)['input_ids']) > 512:
            raise ValueError('質問が長すぎます。512トークン以内に短くしてください')
        with self.lock:
            return self.model.encode(['query: ' + text], normalize_embeddings=True)[0]


class Generator:
    def __init__(self, name, threads=6, adapter_run=''):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        torch.set_num_threads(threads)
        revision = None
        if adapter_run:
            run = Path(adapter_run)
            manifest = json.loads((run / 'run.json').read_text())
            if manifest['config']['model'] != name or manifest.get('engine') != 'pytorch-cpu':
                raise ValueError('RAGと同じベースモデルで学習したPyTorchアダプターを指定してください')
            revision = manifest.get('base_revision')
        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(name, revision=revision,
            torch_dtype=torch.float32, trust_remote_code=False, attn_implementation='eager')
        if adapter_run:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, Path(adapter_run) / 'adapter')
        self.model.eval()
        self.lock = threading.Lock()

    def token_count(self, text):
        return len(self.tokenizer(text, add_special_tokens=False)['input_ids'])

    def generate(self, messages, max_new_tokens):
        import torch
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
        inputs = self.tokenizer(prompt, return_tensors='pt')
        input_size = inputs['input_ids'].shape[1]
        if input_size + max_new_tokens > min(self.model.config.max_position_embeddings, 4096):
            raise ValueError('質問と資料が長すぎます。参照範囲を絞ってください')
        with self.lock, torch.inference_mode():
            result = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                         pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(result[0][input_size:], skip_special_tokens=True)
