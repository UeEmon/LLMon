"""Streamlit entry point for local RAG and reviewed LoRA data preparation."""
import hashlib
import json
import os
from pathlib import Path
import streamlit as st
from rag_documents import extract_document, chunk_sections
from rag_store import Store
from rag_answer import evidence_for_prompt, messages_for, validate_answer

st.title('LLMon - 専門文書RAG')
st.caption('文書を参照して回答。確認済みの質問・回答だけをLoRA追加学習へ。')
from training_manager import Manager, load_rag_config
cfg = load_rag_config(os.environ.get('RAG_CONFIG', 'config/rag.json'), Path.cwd())
if Manager(Path.cwd()).active():
    st.warning('学習または比較ジョブが実行中です。メモリ確保のためRAGは完了まで休止します。学習ジョブ画面で状況を確認してください。')
    st.stop()
store = Store(cfg['database'])


@st.cache_resource
def get_embedder(name, threads):
    from rag_models import Embedder
    return Embedder(name, threads)


@st.cache_resource
def get_generator(name, threads, adapter_run):
    from rag_models import Generator
    return Generator(name, threads, adapter_run)


st.sidebar.header('実行設定')
st.sidebar.write('埋め込みモデル', cfg['embedding_model'])
st.sidebar.write('回答モデル', cfg['answer_model'])
st.sidebar.write('LoRA', cfg['adapter_run'] or '未適用')
st.sidebar.info('推論はDocker内のCPUで実行します。初回はモデル取得が必要です。学習とRAG回答を同時実行しないでください。')
st.sidebar.caption('文書・索引・確認済みデータはdata/privateに保存。外部LLM APIは使用しません。')
tab_docs, tab_query, tab_lora = st.tabs(['1. 文書登録', '2. 質問・出典確認', '3. LoRA用データ'])

with tab_docs:
    st.subheader('PDF / Word / テキストを登録')
    st.info('画像・スキャンPDFは事前にOCRしてください。図の解釈・表の結合セル・脚注などは原本で確認が必要です。')
    uploads = st.file_uploader('専門文書（1ファイル50MBまで）', type=['pdf','docx','txt','md'], accept_multiple_files=True)
    if st.button('選択した文書を登録', disabled=not uploads):
        for upload in uploads:
            try:
                raw = upload.getvalue()
                doc_id = hashlib.sha256(raw).hexdigest()
                if store.has_document(doc_id):
                    st.info(f'{upload.name}: 同一内容は登録済みです')
                    continue
                with st.spinner(f'{upload.name}を抽出・索引化しています'):
                    sections, warnings = extract_document(upload.name, raw)
                    embedder = get_embedder(cfg['embedding_model'], cfg['threads'])
                    chunks = chunk_sections(sections, cfg['chunk_chars'], cfg['chunk_overlap'], embedder.token_count)
                    if len(chunks) > 10000:
                        raise ValueError('チャンクが1万件を超えます。文書を分割してください')
                    vectors = embedder.passages([c['text'] for c in chunks])
                    created = store.add_document(doc_id, upload.name, chunks, vectors, embedder.identity, warnings)
                st.success(f'{upload.name}: {len(chunks)}チャンクを登録しました' if created else '登録済みです')
                for warning in warnings:
                    st.warning(warning)
            except Exception as exc:
                st.error(f'{upload.name}: {exc}')
    docs = store.documents()
    if docs:
        st.dataframe([{'文書名':d['name'], '登録日時':d['created_at'], 'チャンク数':d['chunks'], 'ID':d['id'][:12]} for d in docs],
                     use_container_width=True, hide_index=True)
        for doc in docs:
            with st.expander(f"抽出結果を確認: {doc['name']} ({doc['id'][:8]})"):
                for warning in json.loads(doc['warnings']):
                    st.warning(warning)
                for chunk in store.preview(doc['id']):
                    st.caption(chunk['location'])
                    st.text(chunk['text'])
    else:
        st.caption('文書はまだ登録されていません。まず公開・非機密の短い文書で試してください。')

with tab_query:
    docs = store.documents()
    doc_map = {d['id']:f"{d['name']} ({d['id'][:8]})" for d in docs}
    with st.form('query_form'):
        selected_ids = st.multiselect('参照する文書（未選択なら全登録文書）', options=list(doc_map), format_func=doc_map.get)
        question = st.text_area('専門文書について質問', height=100, max_chars=1500)
        generate_answer = st.checkbox('検索結果から回答も生成する', value=True)
        submit = st.form_submit_button('検索・回答', disabled=not docs)
    if submit:
        if not question.strip():
            st.error('質問を入力してください')
        else:
            try:
                with st.spinner('関連箇所を検索しています'):
                    embedder = get_embedder(cfg['embedding_model'], cfg['threads'])
                    hits = store.search(embedder.query(question), embedder.identity, cfg['top_k'], selected_ids or None)
                result = {'question':question, 'hits':hits, 'evidence':[], 'answer':'', 'warning':None}
                if generate_answer:
                    if not any(h['score'] >= cfg['min_similarity'] for h in hits):
                        result['answer'] = '資料では確認できません。検索結果が設定した類似度の基準を満たしていません。'
                    else:
                        with st.spinner('ローカルモデルが回答を作成しています（CPUのため時間がかかります）'):
                            generator = get_generator(cfg['answer_model'], cfg['threads'], cfg['adapter_run'])
                            evidence = evidence_for_prompt(hits, cfg['min_similarity'], generator.token_count, cfg['context_tokens'])
                            result['evidence'] = evidence
                            if not evidence:
                                result['answer'] = '資料では確認できません。参考箇所が文脈の上限に収まりません。'
                            else:
                                answer = generator.generate(messages_for(question, evidence), cfg['max_new_tokens'])
                                warning = validate_answer(answer, evidence)
                                result['warning'] = warning
                                result['answer'] = answer if warning is None else '生成回答は出典チェックに失敗したため表示していません。原文を確認してください。'
                st.session_state['rag_result'] = result
            except Exception as exc:
                st.session_state.pop('rag_result', None)
                st.error(str(exc))
    result = st.session_state.get('rag_result')
    if result:
        st.subheader('回答')
        st.caption(f"質問: {result['question']}")
        if result['warning']:
            st.error(result['warning'])
        st.text(result['answer'] or '検索のみを実行しました。以下の原文を確認してください。')
        st.warning('類似度は正確さの確率ではありません。出典番号チェックも内容の正しさを保証しません。小型モデルの回答は必ず原文と照合してください。')
        st.subheader('回答に渡した参考資料' if result['evidence'] else '検索結果')
        displayed = result['evidence'] or result['hits']
        for i, hit in enumerate(displayed, 1):
            number = hit.get('citation', i)
            with st.expander(f"[{number}] {hit['name']} / {hit['location']} / 類似度 {hit['score']:.3f}", expanded=True):
                st.text(hit['text'])

with tab_lora:
    st.subheader('回答を人が確認して、学習候補として保存')
    st.write('RAGの索引登録ではモデルの重みは変わりません。LoRAは確認済みデータを別途学習させたときにだけ適用されます。')
    result = st.session_state.get('rag_result')
    if result and result['evidence']:
        with st.form('review_form'):
            reviewed_answer = st.text_area('原文を確認して修正した回答', value=result['answer'], height=180)
            checked = st.checkbox('出典・数値・条件を原本と照合し、学習利用の権限があることを確認しました')
            save = st.form_submit_button('確認済みの質問・回答を保存')
        if save:
            if not checked:
                st.error('確認チェックが必要です')
            else:
                warning = validate_answer(reviewed_answer, result['evidence'])
                if warning:
                    st.error(warning)
                elif reviewed_answer == '生成回答は出典チェックに失敗したため表示していません。原文を確認してください。':
                    st.error('原文に基づく回答へ書き直してください')
                else:
                    source_rows = [{k:h[k] for k in ('document_id','name','location','text','citation')} for h in result['evidence']]
                    store.save_review(result['question'], reviewed_answer, source_rows)
                    st.success('確認済みデータを保存しました。追加学習はまだ実行していません。')
    else:
        st.caption('「質問・出典確認」で参考資料付きの回答を作成すると、ここで確認・保存できます。')
    reviews = store.reviews()
    st.metric('確認済みの学習候補', len(reviews))
    if reviews:
        st.download_button('参考資料付き学習候補をJSONLでダウンロード', store.export_reviews().encode('utf-8'),
                           file_name='reviewed-rag-examples.jsonl', mime='application/jsonl')
        st.caption('このファイルには文書抜粋が含まれます。公開GitHubには登録しないでください。')
        st.info('候補データを文書や章単位でtrain / valid / testへ分割し、系列長を確認してから既存LoRAスクリプトで学習します。詳細はdocs/RAG.mdを参照してください。')
