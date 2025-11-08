# streamlit_app.py
import streamlit as st
from backend.ingest import extract_pages, pages_to_snippets, save_snippets
from backend.embeddings_faiss import FaissIndex
from backend.answer_engine import answer_question
import os

st.set_page_config(page_title='Bajaj AMC - Factsheet RAG Chatbot (Offline)', layout='wide')
st.title('Bajaj AMC Factsheet — Offline RAG Chatbot (Oct 2025)')

if 'faiss' not in st.session_state:
    st.session_state['faiss'] = None
    st.session_state['snippets'] = None
    st.session_state['pages'] = None

with st.sidebar:
    st.header('Ingest Factsheet (Offline)')
    uploaded = st.file_uploader('Upload Bajaj AMC Fund Factsheet (PDF)', type=['pdf'])
    if st.button('Ingest PDF') and uploaded is not None:
        with st.spinner('Extracting pages and building index...'):
            tmp_path = os.path.join('/tmp', uploaded.name)
            with open(tmp_path, 'wb') as f:
                f.write(uploaded.getbuffer())
            pages = extract_pages(tmp_path)
            snippets = pages_to_snippets(pages)
            save_snippets(snippets, '/tmp/snippets')
            faiss = FaissIndex()
            faiss.build(snippets)
            faiss.save('/tmp/faiss_store')
            st.session_state['faiss'] = faiss
            st.session_state['snippets'] = snippets
            st.session_state['pages'] = pages
            st.success(f'Ingested {len(snippets)} snippets from {len(pages)} pages')

    st.markdown('---')
    st.markdown('This app runs entirely locally. No external APIs are called.')

st.write('Ask a question about the uploaded factsheet. The assistant will only use information from the PDF.')
query = st.text_input('Your question')

col1, col2 = st.columns([3,1])
with col2:
    if st.button('Retrieve & Answer'):
        if st.session_state['faiss'] is None:
            st.error('Please ingest the PDF first.')
        elif not query:
            st.error('Enter a question')
        else:
            faiss = st.session_state['faiss']
            retrieved = faiss.query(query, top_k=8)
            answer, citations = answer_question(query, retrieved)
            st.session_state['last_answer'] = answer
            st.session_state['last_citations'] = citations
            st.session_state['last_retrieved'] = retrieved

if 'last_answer' in st.session_state:
    st.subheader('Answer')
    st.write(st.session_state['last_answer'])
    st.markdown('**Citations / Snippets used**')
    for c in st.session_state.get('last_citations', []):
        st.markdown(f"- [Pg {c['page']}] {c['snippet']}")

st.markdown('---')
st.header('Detected Table Snippets (for inspection)')
if st.session_state.get('snippets'):
    snippets = st.session_state['snippets']
    table_snips = [s for s in snippets if s.get('type')=='table' or (',' in s['text'] and '\n' in s['text'])]
    if table_snips:
        for i, s in enumerate(table_snips[:12]):
            with st.expander(f"Page {s['page']} — table snippet {i}"):
                st.text(s['text'])
    else:
        st.info('No table-like snippets detected yet.')
else:
    st.info('Ingest a PDF to enable table detection and direct calculations.')

st.markdown('---')
st.markdown('**Note:** If the engine cannot find a deterministic answer from parsed tables/snippets, it returns \"Not stated in the document.\" or provides the top retrieved snippets for manual inspection.')
