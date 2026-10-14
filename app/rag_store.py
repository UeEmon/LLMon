"""Transactional SQLite document index and human-reviewed LoRA examples."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import numpy as np


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''
              CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, warnings TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP);
              CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY, document_id TEXT REFERENCES documents(id) ON DELETE CASCADE,
                location TEXT NOT NULL, text TEXT NOT NULL, vector BLOB NOT NULL);
              CREATE TABLE IF NOT EXISTS reviewed (
                id TEXT PRIMARY KEY, question TEXT NOT NULL, answer TEXT NOT NULL,
                sources TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
            ''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        try:
            with db:
                yield db
        finally:
            db.close()

    def documents(self):
        with self.connect() as db:
            return [dict(r) for r in db.execute('''SELECT d.*, COUNT(c.id) AS chunks
              FROM documents d LEFT JOIN chunks c ON c.document_id=d.id GROUP BY d.id ORDER BY d.created_at DESC''')]

    def has_document(self, document_id):
        with self.connect() as db:
            return db.execute('SELECT 1 FROM documents WHERE id=?', (document_id,)).fetchone() is not None

    @staticmethod
    def vectors(values):
        vectors = np.asarray(values, dtype='<f4')
        if vectors.ndim != 2 or vectors.shape[1] == 0 or not np.isfinite(vectors).all():
            raise ValueError('埋め込みベクトルが不正です')
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms <= 0):
            raise ValueError('ゼロベクトルは使用できません')
        return vectors / norms

    def add_document(self, document_id, name, chunks, vectors, embedding_id, warnings=None):
        if not chunks:
            raise ValueError('チャンクが空です')
        vectors = self.vectors(vectors)
        if len(chunks) != len(vectors):
            raise ValueError('チャンクとベクトルの数が一致しません')
        signature = json.dumps([embedding_id, vectors.shape[1]])
        with self.connect() as db:
            previous = db.execute("SELECT value FROM metadata WHERE key='embedding'").fetchone()
            if previous and previous['value'] != signature:
                raise ValueError('索引の埋め込みモデルが異なります。別DBで再登録してください')
            if db.execute('SELECT 1 FROM documents WHERE id=?', (document_id,)).fetchone():
                return False
            count = db.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]
            if count + len(chunks) > 10000:
                raise ValueError('索引上限1万チャンクです。文書範囲を絞るか別DBを使用してください')
            db.execute("INSERT OR IGNORE INTO metadata VALUES ('embedding', ?)", (signature,))
            db.execute('INSERT INTO documents (id,name,warnings) VALUES (?,?,?)',
                       (document_id, Path(name).name, json.dumps(warnings or [], ensure_ascii=False)))
            db.executemany('INSERT INTO chunks (document_id,location,text,vector) VALUES (?,?,?,?)',
                [(document_id, c['location'], c['text'], v.astype('<f4').tobytes()) for c,v in zip(chunks,vectors)])
        return True

    def search(self, query_vector, embedding_id, top_k=4, document_ids=None):
        vector = self.vectors([query_vector])[0]
        signature = json.dumps([embedding_id, len(vector)])
        with self.connect() as db:
            previous = db.execute("SELECT value FROM metadata WHERE key='embedding'").fetchone()
            if previous is None:
                return []
            if previous['value'] != signature:
                raise ValueError('索引と検索モデルが一致しません。別DBで文書を再登録してください')
            rows = db.execute('''SELECT c.*, d.name FROM chunks c
                JOIN documents d ON d.id=c.document_id ORDER BY c.id''').fetchall()
        if document_ids is not None:
            rows = [r for r in rows if r['document_id'] in document_ids]
        if not rows:
            return []
        matrix = np.stack([np.frombuffer(r['vector'], dtype='<f4') for r in rows])
        scores = matrix @ vector
        # Diversify near-duplicate overlapping chunks while preserving source locations.
        selected = []
        for idx in np.argsort(-scores):
            row = rows[int(idx)]
            if any(s['document_id'] == row['document_id'] and s['location'] == row['location']
                   and (s['text'] in row['text'] or row['text'] in s['text']) for s in selected):
                continue
            selected.append({k: row[k] for k in ('id','document_id','name','location','text')}
                            | {'score': float(scores[idx])})
            if len(selected) >= top_k:
                break
        return selected

    def preview(self, document_id, limit=8):
        with self.connect() as db:
            return [dict(r) for r in db.execute('SELECT location,text FROM chunks WHERE document_id=? LIMIT ?',
                                               (document_id, limit))]

    def save_review(self, question, answer, sources):
        question, answer = question.strip(), answer.strip()
        if not question or not answer or not sources:
            raise ValueError('質問・回答・出典が必要です')
        payload = json.dumps(sources, ensure_ascii=False, sort_keys=True)
        key = hashlib.sha256((question + '\0' + answer + '\0' + payload).encode()).hexdigest()
        with self.connect() as db:
            db.execute('INSERT OR IGNORE INTO reviewed (id,question,answer,sources) VALUES (?,?,?,?)',
                       (key, question, answer, payload))
        return key

    def reviews(self):
        with self.connect() as db:
            return [dict(r) for r in db.execute('SELECT * FROM reviewed ORDER BY created_at,id')]

    def export_reviews(self, context_in_prompt=True):
        examples = []
        for row in self.reviews():
            sources = json.loads(row['sources'])
            evidence = '\n\n'.join(f"[{i}] {s['name']} / {s['location']}\n{s['text']}"
                                    for i,s in enumerate(sources,1))
            user = f"参考資料:\n{evidence}\n\n質問: {row['question']}" if context_in_prompt else row['question']
            examples.append({'messages': [
                {'role':'system','content':'参考資料を根拠に日本語で回答してください。資料にない事項は断定しないでください。'},
                {'role':'user','content':user}, {'role':'assistant','content':row['answer']}],
                'source_metadata':sources, 'reviewed':True})
        return '\n'.join(json.dumps(row, ensure_ascii=False) for row in examples) + ('\n' if examples else '')
