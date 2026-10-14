"""Unified local learning management interface with native multipage navigation."""
import streamlit as st
from training_ui import overview_page, datasets_page, settings_page, jobs_page, artifacts_page

st.set_page_config(page_title='LLMon | ローカル学習管理',page_icon='📚',layout='wide')
st.sidebar.title('LLMon')
st.sidebar.caption('M4・24GB / ローカル学習管理')
navigation=st.navigation([
    st.Page(overview_page,title='環境概要',icon='🏠',default=True),
    st.Page(datasets_page,title='学習データ',icon='🗂️'),
    st.Page(settings_page,title='学習設定',icon='⚙️'),
    st.Page(jobs_page,title='学習ジョブ・ログ',icon='▶️'),
    st.Page(artifacts_page,title='成果物・回答比較',icon='📦'),
    st.Page('rag_ui.py',title='専門文書RAG',icon='📚')])
st.sidebar.caption('ローカル単一利用者向け。LAN公開・外部API送信は行いません。')
navigation.run()
