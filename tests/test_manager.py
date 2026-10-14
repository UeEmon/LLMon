from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
from training_manager import Manager, validate_config, tail, load_rag_config
from job_worker import run_job

CFG={'model':'Qwen/Qwen3-0.6B','max_steps':20,'max_length':256,'batch_size':1,
     'gradient_accumulation_steps':4,'learning_rate':.0001,'lora_rank':8,'lora_alpha':16,'threads':6,'seed':42}


class ManagerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.manager=Manager(self.root)
        self.files={split:(json.dumps({'messages':[{'role':'user','content':split+' question'},
                       {'role':'assistant','content':'answer'}]})+'\n').encode() for split in ('train','valid','test')}
        self.dataset_id=self.manager.save_dataset('test',self.files)

    def tearDown(self):
        self.tmp.cleanup()

    def test_config_limits_and_unknown_keys(self):
        self.assertEqual(validate_config(CFG),CFG)
        for cfg in (CFG|{'model':'arbitrary'},CFG|{'threads':10},CFG|{'batch_size':2},CFG|{'shell':'rm'}):
            with self.assertRaises(ValueError):
                validate_config(cfg)

    def test_data_registration_and_duplicate_question_validation(self):
        self.assertEqual(json.loads(self.manager.dataset(self.dataset_id)['counts']),{'train':1,'valid':1,'test':1})
        with self.assertRaises(ValueError):
            self.manager.save_dataset('bad',{'train':self.files['train'],'valid':self.files['train'],'test':self.files['test']})
        self.assertEqual(len(self.manager.datasets()),1)

    def test_preset_versioning(self):
        a=self.manager.save_preset('cfg',CFG);b=self.manager.save_preset('cfg',CFG)
        self.assertNotEqual(a,b)
        self.assertEqual(len(self.manager.presets()),2)

    def test_immutable_job_inputs_and_reject_mutated_dataset(self):
        job_id=self.manager.reserve_job('train',CFG,self.dataset_id)
        snap=self.manager.job_dir(job_id)/'data/train.jsonl'
        original=snap.read_bytes()
        source=self.manager.safe_path(self.manager.dataset(self.dataset_id)['path'])/'train.jsonl'
        source.write_text('modified')
        self.assertEqual(snap.read_bytes(),original)
        with self.assertRaisesRegex(ValueError,'変更'):
            self.manager.reserve_job('train',CFG,self.dataset_id)

    def test_single_job_under_concurrent_submission(self):
        def reserve():
            try:
                return self.manager.reserve_job('train',CFG,self.dataset_id)
            except ValueError:
                return None
        with ThreadPoolExecutor(max_workers=2) as pool:
            results=list(pool.map(lambda _:reserve(),range(2)))
        self.assertEqual(sum(r is not None for r in results),1)

    def test_path_traversal_rejected(self):
        with self.assertRaises(ValueError):
            self.manager.safe_path('../../etc/passwd')

    def test_missing_process_becomes_interrupted(self):
        job_id=self.manager.reserve_job('train',CFG,self.dataset_id)
        self.manager.update_job(job_id,state='running',worker_pid=99999999,worker_fingerprint='old',heartbeat=0)
        self.assertEqual(self.manager.jobs()[0]['state'],'interrupted')

    def test_stop_before_worker_start(self):
        job_id=self.manager.reserve_job('train',CFG,self.dataset_id)
        self.manager.request_stop(job_id)
        run_job(self.manager,job_id,lambda manager,job:[([sys.executable,'-c','raise Exception("must not run")'],None)])
        self.assertEqual(self.manager.job(job_id)['state'],'stopped')

    def test_real_child_exit_is_recorded(self):
        job_id=self.manager.reserve_job('train',CFG,self.dataset_id)
        run_job(self.manager,job_id,lambda manager,job:[([sys.executable,'-c','print("test log"); raise SystemExit(7)'],None)])
        job=self.manager.job(job_id)
        self.assertEqual(job['state'],'failed')
        self.assertEqual(job['return_code'],7)
        self.assertIn('test log',tail(self.manager.job_dir(job_id)/'job.log'))

    def test_running_child_can_be_cancelled(self):
        job_id=self.manager.reserve_job('train',CFG,self.dataset_id)
        thread=threading.Thread(target=run_job,args=(self.manager,job_id,
            lambda manager,job:[([sys.executable,'-c','import time; time.sleep(30)'],None)]))
        thread.start()
        deadline=time.monotonic()+4
        while self.manager.job(job_id)['state']=='queued' and time.monotonic()<deadline:
            time.sleep(.02)
        self.manager.request_stop(job_id)
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(self.manager.job(job_id)['state'],'stopped')

    def make_artifact(self):
        run=self.root/'outputs/cpu/finished'
        (run/'adapter').mkdir(parents=True)
        (run/'adapter/adapter_config.json').write_text('{}')
        (run/'run.json').write_text(json.dumps({'engine':'pytorch-cpu','config':CFG}))
        (run/'test_metrics.json').write_text('{"test_loss":1.0}')
        return run

    def test_success_requires_outputs_and_zip_contains_manifest(self):
        self.make_artifact()
        data=self.manager.adapter_zip('outputs/cpu/finished')
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            self.assertIn('run.json',archive.namelist())
            self.assertIn('adapter/adapter_config.json',archive.namelist())
        job_id=self.manager.reserve_job('compare',source_run='outputs/cpu/finished',prompt='question')
        run_job(self.manager,job_id,lambda manager,job:[([sys.executable,'-c','print("answer")'],self.manager.job_dir(job_id)/name)
            for name in ('base.txt','adapted.txt')])
        self.assertEqual(self.manager.job(job_id)['state'],'completed')

    def test_worker_can_only_claim_once(self):
        job_id=self.manager.reserve_job('train',CFG,self.dataset_id)
        self.assertTrue(self.manager.claim_job(job_id))
        self.assertFalse(self.manager.claim_job(job_id))

    def test_private_rag_override_does_not_change_repository_config(self):
        config=self.root/'rag.json';config.write_text('{"adapter_run":"","answer_model":"Qwen/Qwen3-0.6B"}')
        (self.manager.private/'rag-override.json').write_text('{"adapter_run":"outputs/cpu/finished"}')
        self.assertEqual(load_rag_config(config,self.root)['adapter_run'],'outputs/cpu/finished')
        self.assertEqual(json.loads(config.read_text())['adapter_run'],'')


if __name__=='__main__':
    unittest.main()
