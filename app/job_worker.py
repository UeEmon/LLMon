"""Background worker survives browser closure; no shell or Docker socket."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from training_manager import Manager, fingerprint, utcnow


def commands(manager,job):
    app=Path(__file__).parent
    folder=manager.job_dir(job['id'])
    if job['kind']=='train':
        return [([sys.executable,str(app/'train_cpu.py'),'--config',str(folder/'config.json'),
                   '--data',str(folder/'data'),'--output',str(manager.safe_path(job['run_path']))],None)]
    prompt=job['prompt']
    base=[sys.executable,str(app/'infer_cpu.py'),'--run',str(manager.safe_path(job['source_run'])),'--prompt',prompt]
    return [(base+['--base-only'],folder/'base.txt'),(base,folder/'adapted.txt')]


def run_job(manager,job_id,command_builder=commands):
    job=manager.job(job_id)
    if job['state'] not in ('queued','stopping'):
        return
    if not manager.claim_job(job_id):
        return
    folder=manager.job_dir(job_id)
    child=None
    try:
        with (folder/'job.log').open('ab') as log:
            for argv,result_file in command_builder(manager,job):
                if manager.job(job_id)['stop_requested']:
                    manager.update_job(job_id,state='stopped',finished_at=utcnow())
                    return
                log.write((utcnow()+' | 開始: '+Path(argv[1]).name+'\n').encode());log.flush()
                env=os.environ.copy();env['PYTHONUNBUFFERED']='1'
                output=result_file.open('wb') if result_file else None
                try:
                    child=subprocess.Popen(argv,cwd=manager.root,stdout=output or log,stderr=log,
                                           env=env,start_new_session=True)
                    stop_at=None
                    while child.poll() is None:
                        manager.update_job(job_id,heartbeat=time.time())
                        if manager.job(job_id)['stop_requested']:
                            if stop_at is None:
                                try:
                                    os.killpg(child.pid,signal.SIGTERM)
                                except ProcessLookupError:
                                    pass
                                stop_at=time.monotonic()
                            elif time.monotonic()-stop_at>8:
                                os.killpg(child.pid,signal.SIGKILL)
                        time.sleep(.5)
                    code=child.returncode
                    if manager.job(job_id)['stop_requested']:
                        manager.update_job(job_id,state='stopped',return_code=code,finished_at=utcnow())
                        return
                    if code!=0:
                        manager.update_job(job_id,state='failed',return_code=code,finished_at=utcnow(),error='処理がエラー終了しました。ログを確認してください')
                        return
                finally:
                    if output:
                        output.close()
            if job['kind']=='train':
                run=manager.safe_path(job['run_path'])
                if not (run/'adapter/adapter_config.json').exists() or not (run/'test_metrics.json').exists():
                    raise ValueError('終了コードは0ですが必要な成果物がありません')
            elif not all((folder/name).exists() and (folder/name).stat().st_size for name in ('base.txt','adapted.txt')):
                raise ValueError('学習前と学習後の回答ファイルが揃っていません')
            manager.update_job(job_id,state='completed',return_code=0,finished_at=utcnow())
    except Exception as exc:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
        manager.update_job(job_id,state='failed',finished_at=utcnow(),error=str(exc))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--job-id',required=True)
    args=parser.parse_args()
    run_job(Manager(args.root),args.job_id)
