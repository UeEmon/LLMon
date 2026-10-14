"""Web pages for datasets, guarded jobs, progress and LoRA artifacts."""
import gc
import json
import os
from pathlib import Path
import platform
import shutil
import streamlit as st
from training_manager import Manager, MODELS, tail, load_rag_config

LABELS={'queued':'開始待ち','running':'実行中','stopping':'停止処理中','stopped':'停止済み',
        'completed':'完了','failed':'失敗','interrupted':'中断'}


def manager():
    return Manager(Path.cwd())


def launch_job(control,job_id):
    # Free cached RAG models before launching a separate training process.
    st.cache_resource.clear()
    gc.collect()
    control.launch(job_id)
    st.session_state['selected_job']=job_id


def overview_page():
    st.title('ローカル学習環境')
    st.caption('文書を参照するRAGと、確認済みデータによるLoRA学習を分けて管理します。')
    control=manager()
    active=control.active()
    cols=st.columns(3)
    cols[0].metric('登録データセット',len(control.datasets()))
    cols[1].metric('実行中のジョブ',len(active))
    cols[2].metric('PyTorchアダプター',len(control.artifacts()))
    st.subheader('最初の試行')
    st.write('1. 学習データで付属サンプルを登録 → 2. 学習設定を保存 → 3. 学習ジョブで開始 → 4. 成果物で回答を比較')
    st.info('Webから開始・停止できるのはDocker内のCPU LoRAと回答比較です。M4 GPUのMLX学習はMacのターミナルで実行します。')
    disk=shutil.disk_usage(Path.cwd())
    st.json({'実行環境':platform.platform(),'CPU論理コア（コンテナから見える数）':os.cpu_count(),
             '処理': 'PyTorch CPU / Metal利用なし','空きディスクGB':round(disk.free/1024**3,1),
             'Web接続先':'http://localhost:18083','JupyterLab':'http://localhost:18082'})
    st.subheader('モデルキャッシュの状態')
    cache=Path(os.environ.get('HF_HOME',str(Path.home()/'.cache/huggingface')))/'hub'
    st.dataframe([{'モデル':name,'スナップショットあり':any((cache/('models--'+name.replace('/','--'))/'snapshots').glob('*'))}
                  for name in (*MODELS,'intfloat/multilingual-e5-small')],hide_index=True,use_container_width=True)
    st.caption('スナップショットの存在表示は、モデルの全ファイル取得・正常動作を保証しません。初回学習・RAG時に不足分を取得します。')
    with st.expander('M4 GPUでのMLX学習'):
        st.code('docker compose stop rag\nbash scripts/Setup-MLX.sh\nbash scripts/Train-MLX.sh',language='bash')
        st.caption('MLXの開始・停止をWebから制御するホスト連接は未実装です。MacのGPUはDockerへ直接公開しません。')


def datasets_page():
    st.title('学習データ')
    control=manager()
    st.write('train / valid / testを別ファイルで登録します。登録ごとに別の版を保存し、既存データを上書きしません。')
    if st.button('付属サンプルを登録（24 / 4 / 4件）'):
        try:
            control.import_sample();st.success('付属サンプルを登録しました')
        except Exception as exc:
            st.error(str(exc))
    with st.form('dataset_upload'):
        name=st.text_input('データセット名',max_chars=120)
        files={split:st.file_uploader(f'{split}.jsonl（UTF-8、10MB以下）',type=['jsonl'],key=f'upload_{split}')
               for split in ('train','valid','test')}
        save=st.form_submit_button('検証して新しい版を登録')
    if save:
        try:
            if any(file is None for file in files.values()):
                raise ValueError('3ファイルすべてを選択してください')
            control.save_dataset(name,{split:file.getvalue() for split,file in files.items()})
            st.success('形式・質問重複・元文書の分割漏洩を検証して保存しました')
        except Exception as exc:
            st.error(str(exc))
    datasets=control.datasets()
    if datasets:
        st.dataframe([{'名称':d['name'],'版':d['id'][:8],**json.loads(d['counts']),'登録日時（UTC）':d['created_at']} for d in datasets],
                     hide_index=True,use_container_width=True)
        chosen=st.selectbox('内容を確認する版',datasets,format_func=lambda d:f"{d['name']} / {d['id'][:8]}")
        for split in ('train','valid','test'):
            with st.expander(f'{split}: 先頭3件'):
                path=control.safe_path(chosen['path'])/f'{split}.jsonl'
                with path.open(encoding='utf-8') as handle:
                    for _,line in zip(range(3),(line for line in handle if line.strip())):
                        st.json(json.loads(line))
                st.download_button(f'{split}を取得',path.read_bytes(),file_name=f'{split}.jsonl',key=f'download_{split}')
    st.warning('学習例・文書抜粋を公開GitHubへ登録しないでください。同じ元文書を学習用と評価用へ分けることも禁止しています。')


def settings_page():
    st.title('学習設定')
    control=manager()
    presets=control.presets()
    options=[{'name':'初期設定（短い例文用）','config':(Path('config/cpu.json')).read_text()},
             {'name':'RAG回答用（系列長2048）','config':(Path('config/cpu-rag.json')).read_text()}]+presets
    chosen=st.selectbox('元にする設定',options,format_func=lambda p:p['name'])
    defaults=json.loads(chosen['config'])
    # Distinct form keys let a preset change populate the controls.
    identity=chosen.get('id',chosen['name'])
    with st.form('preset_'+identity):
        name=st.text_input('保存する設定名',value=chosen['name'],max_chars=120)
        model=st.selectbox('ベースモデル',MODELS,index=MODELS.index(defaults['model']))
        left,right=st.columns(2)
        steps=left.number_input('更新ステップ数',1,2000,defaults['max_steps'])
        length=right.number_input('最大系列長（トークン）',128,2048,defaults['max_length'],step=128)
        lr=left.number_input('学習率',min_value=0.000001,max_value=0.0005,value=float(defaults['learning_rate']),format='%.6f')
        rank=right.selectbox('LoRA rank',[4,8,16],index=[4,8,16].index(defaults['lora_rank']))
        alpha=left.number_input('LoRA alpha',1,64,defaults['lora_alpha'])
        accumulation=right.number_input('勾配蓄積回数',1,16,defaults['gradient_accumulation_steps'])
        threads=left.number_input('CPUスレッド数',1,6,defaults['threads'])
        seed=right.number_input('乱数シード',0,2147483647,defaults['seed'])
        save=st.form_submit_button('設定を新しい版として保存')
    if save:
        cfg={'model':model,'max_steps':int(steps),'max_length':int(length),'batch_size':1,
             'learning_rate':float(lr),'lora_rank':rank,'lora_alpha':int(alpha),
             'gradient_accumulation_steps':int(accumulation),'threads':int(threads),'seed':int(seed)}
        try:
            control.save_preset(name,cfg);st.success('設定を保存しました。学習ジョブ画面で選択してください。')
        except Exception as exc:
            st.error(str(exc))
    st.info('M4・24GB向けにバッチサイズは1固定です。初回は0.6B・20ステップ・系列長256で確認してください。1.7Bや長い系列ではメモリ不足・長時間処理の可能性があります。')


def jobs_page():
    st.title('学習ジョブ・ログ')
    control=manager()
    datasets,presets=control.datasets(),control.presets()
    active=control.active()
    if datasets and presets:
        with st.form('start_training'):
            dataset=st.selectbox('データセット',datasets,format_func=lambda d:f"{d['name']} / {d['id'][:8]}")
            preset=st.selectbox('学習設定',presets,format_func=lambda p:f"{p['name']} / {p['id'][:8]}")
            confirmed=st.checkbox('CPUで実行することと、学習データ利用の権限を確認しました')
            start=st.form_submit_button('LoRA学習を開始',disabled=bool(active))
        if start:
            try:
                if not confirmed:
                    raise ValueError('確認チェックを入れてください')
                job_id=control.reserve_job('train',json.loads(preset['config']),dataset['id'])
                launch_job(control,job_id)
                st.success('バックグラウンド学習を開始しました。ブラウザを閉じても処理は続きます。')
            except Exception as exc:
                st.error(str(exc))
    else:
        st.info('学習データと学習設定を先に保存してください。')
    st.caption('ジョブは同時に1件だけ実行します。開始時にデータと設定のコピーを保存し、実行中の編集による変更を防ぎます。')
    st.button('状態を更新')
    jobs=control.jobs()
    if not jobs:
        return
    st.dataframe([{'ID':j['id'][:8],'種別':'LoRA学習' if j['kind']=='train' else '回答比較',
                  '状態':LABELS[j['state']],'開始日時（UTC）':j['created_at'],'終了コード':j['return_code']} for j in jobs],
                 hide_index=True,use_container_width=True)
    wanted=st.session_state.get('selected_job')
    index=next((i for i,j in enumerate(jobs) if j['id']==wanted),0)
    selected=st.selectbox('詳細を表示',jobs,index=index,format_func=lambda j:f"{j['id'][:8]} / {LABELS[j['state']]}")
    st.subheader(LABELS[selected['state']])
    if selected['error']:
        st.error(selected['error'])
    folder=control.job_dir(selected['id'])
    if selected['kind']=='train':
        records=[]
        for line in tail(control.safe_path(selected['run_path'])/'metrics.jsonl').splitlines():
            try:
                records.append(json.loads(line))
            except ValueError:
                pass
        if records:
            latest=records[-1]
            st.progress(min(1.0,latest['step']/max(1,latest['max_steps'])),text=f"更新ステップ {latest['step']} / {latest['max_steps']}")
            st.dataframe(records[-100:],hide_index=True,use_container_width=True)
        else:
            st.caption('モデル取得・データ検証・初期化中は進捗ステップがまだ表示されません。ログを確認してください。')
    if selected['state'] in ('queued','running','stopping'):
        stop_confirm=st.checkbox('このジョブを停止する（途中成果物は完全なモデルとは限りません）',key='stop_'+selected['id'])
        if st.button('停止を要求',disabled=not stop_confirm):
            control.request_stop(selected['id']);st.warning('停止を要求しました。状態更新で完了を確認してください。')
    st.text_area('実行ログ（末尾64KB）',value=tail(folder/'job.log') or tail(folder/'worker.log'),height=300,disabled=True)
    st.caption('停止・失敗時もログと作成済みチェックポイントを残します。自動再開はありません。')


def artifacts_page():
    st.title('成果物・回答比較')
    control=manager()
    artifacts=control.artifacts()
    if not artifacts:
        st.info('学習済みPyTorchアダプターはまだありません。')
    else:
        chosen=st.selectbox('学習成果物',artifacts,format_func=lambda a:a['path'])
        st.json(chosen['manifest'])
        if chosen['metrics']:
            st.subheader('未使用testデータでの評価');st.json(chosen['metrics'])
        else:
            st.warning('評価結果がありません。途中または評価に失敗した成果物の可能性があります。ログを確認してください。')
        if st.button('アダプターZIPを準備'):
            try:
                st.session_state['adapter_zip']=(chosen['path'],control.adapter_zip(chosen['path']))
            except Exception as exc:
                st.error(str(exc))
        saved=st.session_state.get('adapter_zip')
        if saved and saved[0]==chosen['path']:
            st.download_button('アダプター・設定・評価結果を取得',saved[1],file_name=Path(chosen['path']).name+'-adapter.zip')
            st.caption('ZIPにベースモデル本体は含みません。モデル名と取得revisionはrun.jsonに記録されています。')
        with st.form('compare_form'):
            prompt=st.text_area('学習に使っていない質問で回答を比較',max_chars=1500)
            compare=st.form_submit_button('学習前／学習後の比較を開始',disabled=bool(control.active()))
        if compare:
            try:
                job_id=control.reserve_job('compare',source_run=chosen['path'],prompt=prompt)
                launch_job(control,job_id);st.success('回答比較を開始しました。学習ジョブ画面で状態を更新してください。')
            except Exception as exc:
                st.error(str(exc))
        if chosen['manifest']['config']['model']=='Qwen/Qwen3-0.6B' and chosen['metrics']:
            apply=st.checkbox('この成果物をRAGへ適用する（設定ファイルを更新）')
            if st.button('RAGへ適用',disabled=not apply or bool(control.active())):
                config_path=Path(os.environ.get('RAG_CONFIG','config/rag.json'))
                config=load_rag_config(config_path,Path.cwd())
                if config['answer_model']!=chosen['manifest']['config']['model']:
                    st.error('RAGとベースモデルが一致していません')
                else:
                    override=control.private/'rag-override.json'
                    temporary=override.with_suffix('.json.tmp')
                    temporary.write_text(json.dumps({'adapter_run':chosen['path']},ensure_ascii=False,indent=2)+'\n')
                    temporary.replace(override)
                    st.cache_resource.clear();gc.collect()
                    st.success('適用設定を保存しました。専門文書RAG画面で確認してください。')
        reset=st.checkbox('RAGのLoRA適用を解除してベースモデルへ戻す')
        if st.button('RAGの適用を解除',disabled=not reset or bool(control.active())):
            override=control.private/'rag-override.json'
            override.write_text(json.dumps({'adapter_run':''})+'\n')
            st.cache_resource.clear();gc.collect()
            st.success('LoRAの適用を解除しました。学習成果物はそのまま残しています。')
        for job in control.jobs():
            if job['kind']=='compare' and job['source_run']==chosen['path'] and job['state']=='completed':
                folder=control.job_dir(job['id'])
                st.subheader(f"比較結果: {job['id'][:8]}");st.text(job['prompt'])
                left,right=st.columns(2)
                left.caption('学習前');left.text(tail(folder/'base.txt'))
                right.caption('学習後');right.text(tail(folder/'adapted.txt'))
    with st.expander('Mac側MLX成果物（参照のみ）'):
        runs=[str(p) for p in Path('outputs/mlx').glob('*/adapter') if p.is_dir()]
        st.write(runs or 'MLX成果物はありません')
        st.caption('MLXは別形式です。Docker内RAGへの直接適用、Webからの開始・停止は未対応です。')
