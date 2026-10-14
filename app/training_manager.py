"""Single-user training control plane; private immutable inputs and durable jobs."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile
from data_utils import validate_directory

MODELS = ('Qwen/Qwen3-0.6B', 'Qwen/Qwen3-1.7B')
ACTIVE = ('queued', 'running', 'stopping')
CONFIG_KEYS = {'model','max_length','max_steps','batch_size','gradient_accumulation_steps',
               'learning_rate','lora_rank','lora_alpha','threads','seed'}


def validate_config(cfg):
    if set(cfg) != CONFIG_KEYS or cfg['model'] not in MODELS:
        raise ValueError('設定項目またはモデルが未対応です')
    bounds = {'max_length':(128,2048),'max_steps':(1,2000),'batch_size':(1,1),
              'gradient_accumulation_steps':(1,16),'lora_rank':(4,16),'lora_alpha':(1,64),
              'threads':(1,6),'seed':(0,2147483647)}
    for key,(low,high) in bounds.items():
        if type(cfg[key]) is not int or not low <= cfg[key] <= high:
            raise ValueError(f'{key}: {low}〜{high}の整数が必要です')
    if cfg['lora_rank'] not in (4,8,16):
        raise ValueError('LoRA rankは4/8/16から選択してください')
    lr = cfg['learning_rate']
    if isinstance(lr, bool) or not isinstance(lr,(float,int)) or not 1e-6 <= lr <= 5e-4:
        raise ValueError('learning_rateは0.000001〜0.0005です')
    return dict(cfg)


def utcnow():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def fingerprint(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text()
        state_and_fields = stat[stat.rfind(')')+2:].split()
        if state_and_fields[0] == 'Z':
            return None
        return Path('/proc/sys/kernel/random/boot_id').read_text().strip() + ':' + state_and_fields[19]
    except (OSError, IndexError):
        return None


def tail(path, limit=64000):
    try:
        with Path(path).open('rb') as handle:
            handle.seek(0,2)
            size = handle.tell()
            handle.seek(max(0,size-limit))
            return handle.read().decode('utf-8',errors='replace')
    except FileNotFoundError:
        return ''


def load_rag_config(config_path, root):
    cfg=json.loads(Path(config_path).read_text())
    override=Path(root)/'data/private/manager/rag-override.json'
    if override.exists():
        cfg['adapter_run']=json.loads(override.read_text())['adapter_run']
    return cfg


class Manager:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.private = self.root / 'data/private/manager'
        self.private.mkdir(parents=True,exist_ok=True)
        self.db_path = self.private / 'state.sqlite3'
        with self.connect() as db:
            db.executescript('''
              CREATE TABLE IF NOT EXISTS datasets(id TEXT PRIMARY KEY, name TEXT, path TEXT,
                  counts TEXT, sha256 TEXT, created_at TEXT);
              CREATE TABLE IF NOT EXISTS presets(id TEXT PRIMARY KEY, name TEXT, config TEXT, created_at TEXT);
              CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, kind TEXT, state TEXT,
                  config TEXT, dataset_id TEXT, run_path TEXT, source_run TEXT, prompt TEXT,
                  worker_pid INTEGER, worker_fingerprint TEXT, stop_requested INTEGER DEFAULT 0,
                  created_at TEXT, heartbeat REAL, finished_at TEXT, return_code INTEGER, error TEXT);
            ''')

    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.db_path,timeout=30)
        db.row_factory=sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def safe_path(self, relative):
        path=(self.root/relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('プロジェクト外のパスは使用できません')
        return path

    def save_dataset(self, name, files):
        if not name.strip() or len(name)>120 or set(files) != {'train','valid','test'}:
            raise ValueError('名称とtrain / valid / testの3ファイルが必要です')
        dataset_id=uuid.uuid4().hex
        destination=self.private/'datasets'/dataset_id
        destination.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination.parent,prefix='staging-') as tmp:
            stage=Path(tmp)
            hashes={}
            for split,raw in files.items():
                if not isinstance(raw,bytes) or not 0 < len(raw) <= 10*1024*1024:
                    raise ValueError('各JSONLは空でない10MB以下のファイルにしてください')
                raw.decode('utf-8-sig')
                (stage/f'{split}.jsonl').write_bytes(raw[3:] if raw.startswith(b'\xef\xbb\xbf') else raw)
                hashes[split]=hashlib.sha256((stage/f'{split}.jsonl').read_bytes()).hexdigest()
            splits=validate_directory(stage)
            shutil.copytree(stage,destination)
        counts={key:len(rows) for key,rows in splits.items()}
        with self.connect() as db:
            db.execute('INSERT INTO datasets VALUES (?,?,?,?,?,?)',
                (dataset_id,name.strip(),str(destination.relative_to(self.root)),json.dumps(counts),json.dumps(hashes),utcnow()))
        return dataset_id

    def import_sample(self):
        return self.save_dataset('付属の業務連絡サンプル',
            {split:(self.root/'data'/f'{split}.jsonl').read_bytes() for split in ('train','valid','test')})

    def datasets(self):
        with self.connect() as db:
            return [dict(row) for row in db.execute('SELECT * FROM datasets ORDER BY created_at DESC,id')]

    def dataset(self,dataset_id):
        with self.connect() as db:
            row=db.execute('SELECT * FROM datasets WHERE id=?',(dataset_id,)).fetchone()
        if row is None:
            raise ValueError('データセットがありません')
        return dict(row)

    def save_preset(self,name,cfg):
        cfg=validate_config(cfg)
        if not name.strip() or len(name)>120:
            raise ValueError('設定名を入力してください（120文字以内）')
        preset_id=uuid.uuid4().hex
        with self.connect() as db:
            db.execute('INSERT INTO presets VALUES (?,?,?,?)',(preset_id,name.strip(),json.dumps(cfg),utcnow()))
        return preset_id

    def presets(self):
        with self.connect() as db:
            return [dict(row) for row in db.execute('SELECT * FROM presets ORDER BY created_at DESC,id')]

    def jobs(self):
        self.reconcile()
        with self.connect() as db:
            return [dict(row) for row in db.execute('SELECT * FROM jobs ORDER BY created_at DESC,id')]

    def job(self,job_id):
        with self.connect() as db:
            row=db.execute('SELECT * FROM jobs WHERE id=?',(job_id,)).fetchone()
        if row is None:
            raise ValueError('ジョブがありません')
        return dict(row)

    def update_job(self,job_id,**values):
        allowed={'state','worker_pid','worker_fingerprint','heartbeat','finished_at','return_code','error'}
        if not values or not set(values)<=allowed:
            raise ValueError('更新項目が不正です')
        with self.connect() as db:
            db.execute('UPDATE jobs SET '+','.join(f'{k}=?' for k in values)+' WHERE id=?',(*values.values(),job_id))

    def reconcile(self):
        with self.connect() as db:
            rows=db.execute("SELECT * FROM jobs WHERE state IN ('queued','running','stopping')").fetchall()
            for row in rows:
                if time.time()-(row['heartbeat'] or 0)<30:
                    continue
                if not row['worker_pid'] or fingerprint(row['worker_pid']) != row['worker_fingerprint']:
                    db.execute("UPDATE jobs SET state='interrupted',finished_at=?,error=? WHERE id=?",
                               (utcnow(),'ワーカープロセスが終了しています。ログ・チェックポイントを確認してください',row['id']))

    def active(self):
        return [job for job in self.jobs() if job['state'] in ACTIVE]

    def reserve_job(self,kind,cfg=None,dataset_id=None,source_run='',prompt=''):
        if kind not in ('train','compare'):
            raise ValueError('ジョブ種別が未対応です')
        if kind=='train':
            cfg=validate_config(cfg)
            dataset=self.dataset(dataset_id)
            source=self.safe_path(dataset['path'])
            actual={s:hashlib.sha256((source/f'{s}.jsonl').read_bytes()).hexdigest() for s in ('train','valid','test')}
            if actual != json.loads(dataset['sha256']):
                raise ValueError('登録後にデータが変更されています。新しい版として登録してください')
            validate_directory(source)
        else:
            source=self.safe_path(source_run)
            if not source.is_relative_to(self.root/'outputs/cpu') or not (source/'adapter/adapter_config.json').is_file():
                raise ValueError('PyTorchの学習済みアダプターを選択してください')
            manifest=json.loads((source/'run.json').read_text())
            validate_config(manifest['config'])
            if not prompt.strip() or len(prompt)>1500:
                raise ValueError('比較する質問を1〜1500文字で入力してください')
        self.reconcile()
        job_id=uuid.uuid4().hex
        job_dir=self.root/'outputs/jobs'/job_id
        run_path=str((self.root/'outputs/cpu'/job_id).relative_to(self.root)) if kind=='train' else ''
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute("SELECT 1 FROM jobs WHERE state IN ('queued','running','stopping')").fetchone():
                raise ValueError('別のジョブが実行中です。完了または停止後に開始してください')
            if kind=='train':
                job_dir.mkdir(parents=True)
                shutil.copytree(source,job_dir/'data')
                (job_dir/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
            else:
                job_dir.mkdir(parents=True)
            db.execute('''INSERT INTO jobs (id,kind,state,config,dataset_id,run_path,source_run,prompt,created_at,heartbeat)
                          VALUES (?,?,?,?,?,?,?,?,?,?)''',
                       (job_id,kind,'queued',json.dumps(cfg),dataset_id,run_path,source_run,prompt,utcnow(),time.time()))
        return job_id

    def launch(self,job_id):
        job=self.job(job_id)
        if job['state']!='queued':
            raise ValueError('開始できる状態ではありません')
        log=self.root/'outputs/jobs'/job_id/'worker.log'
        try:
            with log.open('ab') as handle:
                proc=subprocess.Popen([sys.executable,str(Path(__file__).with_name('job_worker.py')),
                    '--root',str(self.root),'--job-id',job_id],cwd=self.root,stdout=handle,stderr=subprocess.STDOUT,
                    start_new_session=True)
            self.update_job(job_id,worker_pid=proc.pid,worker_fingerprint=fingerprint(proc.pid))
        except OSError as exc:
            self.update_job(job_id,state='failed',error=str(exc),finished_at=utcnow())
            raise

    def request_stop(self,job_id):
        with self.connect() as db:
            db.execute("UPDATE jobs SET stop_requested=1,state='stopping' WHERE id=? AND state IN ('queued','running','stopping')",(job_id,))

    def claim_job(self,job_id):
        with self.connect() as db:
            cursor=db.execute("""UPDATE jobs SET state='running',worker_pid=?,worker_fingerprint=?,heartbeat=?
                WHERE id=? AND state IN ('queued','stopping') AND worker_pid IS NULL""",
                (os.getpid(),fingerprint(os.getpid()),time.time(),job_id))
            if cursor.rowcount:
                return True
            # launch() may already have persisted this worker's PID.
            cursor=db.execute("""UPDATE jobs SET state='running',heartbeat=? WHERE id=?
                AND state IN ('queued','stopping') AND worker_pid=? AND worker_fingerprint=?""",
                (time.time(),job_id,os.getpid(),fingerprint(os.getpid())))
            return cursor.rowcount==1

    def job_dir(self,job_id):
        self.job(job_id)
        return self.root/'outputs/jobs'/job_id

    def artifacts(self):
        rows=[]
        base=self.root/'outputs/cpu'
        for folder in sorted(base.glob('*'),reverse=True):
            if not folder.is_dir() or not folder.resolve().is_relative_to(base.resolve()):
                continue
            if (folder/'adapter/adapter_config.json').is_file() and (folder/'run.json').is_file():
                try:
                    manifest=json.loads((folder/'run.json').read_text())
                    if manifest.get('engine')!='pytorch-cpu':
                        continue
                    rows.append({'path':str(folder.relative_to(self.root)), 'manifest':manifest,
                        'metrics':json.loads((folder/'test_metrics.json').read_text()) if (folder/'test_metrics.json').exists() else {}})
                except (OSError,ValueError):
                    continue
        return rows

    def adapter_zip(self,run_path):
        path=self.safe_path(run_path)
        if run_path not in {row['path'] for row in self.artifacts()}:
            raise ValueError('学習成果物を選択してください')
        files=[p for p in (path/'adapter').rglob('*') if p.is_file() and p.resolve().is_relative_to(path.resolve())]
        files += [p for p in (path/'run.json',path/'test_metrics.json') if p.exists()]
        if sum(p.stat().st_size for p in files)>100*1024*1024:
            raise ValueError('100MBを超えます。outputsフォルダから直接取得してください')
        buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
            for file in files:
                archive.write(file,str(file.relative_to(path)))
        return buffer.getvalue()
