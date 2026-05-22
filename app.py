import os
import streamlit as st
import re
import json
import hashlib
import time
import numpy as np
import requests
import faiss
from pypdf import PdfReader
from openai import OpenAI

# 1. ESSENTIAL CONFIG
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
os.environ["STREAMLIT_SERVER_PORT"] = "8080"

# 2. SECRETS & CLIENT INITIALIZATION
def get_secret(key): return os.environ.get(key)
OPENROUTER_API_KEY = get_secret("OPENROUTER_API_KEY")
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
ADMIN_PASSWORD = get_secret("ADMIN_PASSWORD")

if not all([OPENROUTER_API_KEY, OPENAI_API_KEY, ADMIN_PASSWORD]):
    st.error("Missing credentials in Environment Variables.")
    st.stop()

openai_client = OpenAI(api_key=OPENAI_API_KEY)
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# 3. PATHS
INDEX_PATH, METADATA_PATH, CACHE_PATH = "faiss_index.bin", "faiss_metadata.json", "embedding_cache.json"

# 4. PAGE CONFIGURATION
st.set_page_config(page_title="Otimo Aero AI Technician", page_icon="✈️", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>[data-testid='stSidebar'] { display: none !important; }</style>", unsafe_allow_html=True)

# 5. VECTOR FUNCTIONS
def get_embeddings_batched(texts, model="text-embedding-3-small"):
    if "embed_cache" not in st.session_state: st.session_state.embed_cache = json.load(open(CACHE_PATH, "r")) if os.path.exists(CACHE_PATH) else {}
    results, uncached, indices = [None] * len(texts), [], []
    for i, t in enumerate(texts):
        h = hashlib.sha256(t.encode()).hexdigest()
        if h in st.session_state.embed_cache: results[i] = st.session_state.embed_cache[h]
        else: uncached.append(t); indices.append(i)
    if uncached:
        res = openai_client.embeddings.create(input=[t.replace("\n", " ") for t in uncached], model=model)
        for t, d in zip(uncached, res.data):
            st.session_state.embed_cache[hashlib.sha256(t.encode()).hexdigest()] = d.embedding
            results[indices[uncached.index(t)]] = d.embedding
        json.dump(st.session_state.embed_cache, open(CACHE_PATH, "w"))
    return results

def parse_and_chunk_pdf(uploaded_files):
    all_chunks = []
    for f in uploaded_files:
        reader = PdfReader(f)
        for i, page in enumerate(reader.pages):
            if text := page.extract_text(): all_chunks.append({"text": re.sub(r'\s+', ' ', text).strip(), "source": f.name, "page": i + 1})
    if all_chunks:
        embeddings = np.array(get_embeddings_batched([c["text"] for c in all_chunks])).astype('float32')
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(len(embeddings[0])); index.add(embeddings)
        faiss.write_index(index, INDEX_PATH)
        json.dump(all_chunks, open(METADATA_PATH, "w"))
        st.rerun()

# 6. LLM & PROMPT HUB
def call_llm(context, user_query):
    payload = {"model": "meta-llama/llama-3.1-8b-instruct", "temperature": 0.1, "messages": [
        {"role": "system", "content": f"You are 'Otimo Inspector'. Context: {context}"},
        {"role": "user", "content": user_query}]}
    return requests.post(OPENROUTER_URL, headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"}, json=payload).json()["choices"][0]["message"]["content"]

# 7. MAIN UI
col_layout = st.columns([0.15, 0.70, 0.15])[1]
with col_layout:
    st.title("Otimo Aero AI Technician")
    if "messages" not in st.session_state: st.session_state.messages = []
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.write(msg["content"])
    
    if query := st.chat_input("Enter maintenance question..."):
        st.session_state.messages.append({"role": "user", "content": query})
        # Logic for RAG lookup and LLM call here
        st.rerun()

# 8. ADMIN ACCESS
if st.query_params.get("admin") == "true":
    st.markdown("<style>[data-testid='stSidebar'] { display: block !important; }</style>", unsafe_allow_html=True)
    with st.sidebar:
        if st.text_input("Password", type="password") == ADMIN_PASSWORD:
            if files := st.file_uploader("Upload Manuals", accept_multiple_files=True): parse_and_chunk_pdf(files)