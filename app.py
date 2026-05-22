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

# 2. SECRETS & CLIENT
def get_secret(key): return os.environ.get(key)
OPENROUTER_API_KEY = get_secret("OPENROUTER_API_KEY")
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
ADMIN_PASSWORD = get_secret("ADMIN_PASSWORD")

if not OPENROUTER_API_KEY or not OPENAI_API_KEY or not ADMIN_PASSWORD:
    st.error("Missing credentials in Environment Variables.")
    st.stop()

openai_client = OpenAI(api_key=OPENAI_API_KEY)
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
INDEX_PATH, METADATA_PATH, CACHE_PATH = "faiss_index.bin", "faiss_metadata.json", "embedding_cache.json"

# 3. PAGE CONFIG (Only called once)
st.set_page_config(page_title="Otimo Aero AI", page_icon="✈️", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>[data-testid='stSidebar'] { display: none !important; }</style>", unsafe_allow_html=True)

# 4. MASTER SPEC REGISTRY
SPEC_REGISTRY = {
    "OIL CHANGE / MAGNETIC PLUG INSPECTION": {
        "reasoning": ["Warm oil scavenges particulates.", "Torque for tightening only.", "Tapered seat: No sealant."],
        "specs": "- Sump: 25 Nm | Mag Plug: 20 Nm | Filter: Hand-tight + 3/4 turn."
    },
    "SPARK PLUG INSPECTION": {
        "reasoning": ["Cold engine installation.", "0.8-0.9mm gap."],
        "specs": "- Torque: 16 Nm."
    }
}

# 5. VECTOR FUNCTIONS
def get_embeddings(texts):
    if "embed_cache" not in st.session_state: st.session_state.embed_cache = json.load(open(CACHE_PATH, "r")) if os.path.exists(CACHE_PATH) else {}
    results, uncached = [None] * len(texts), []
    for i, t in enumerate(texts):
        h = hashlib.sha256(t.encode()).hexdigest()
        if h in st.session_state.embed_cache: results[i] = st.session_state.embed_cache[h]
        else: uncached.append((i, t))
    if uncached:
        batch = [t for i, t in uncached]
        res = openai_client.embeddings.create(input=batch, model="text-embedding-3-small")
        for (i, t), d in zip(uncached, res.data):
            h = hashlib.sha256(t.encode()).hexdigest()
            st.session_state.embed_cache[h] = d.embedding
            results[i] = d.embedding
        json.dump(st.session_state.embed_cache, open(CACHE_PATH, "w"))
    return results

def search_index(query):
    if "vector_index" not in st.session_state and os.path.exists(INDEX_PATH):
        st.session_state.vector_index = faiss.read_index(INDEX_PATH)
        st.session_state.vector_metadata = json.load(open(METADATA_PATH, "r"))
    
    if "vector_index" in st.session_state:
        q_vec = np.array(get_embeddings([query])).astype('float32')
        faiss.normalize_L2(q_vec)
        _, idxs = st.session_state.vector_index.search(q_vec, 3)
        return "\n".join([st.session_state.vector_metadata[i]['text'] for i in idxs[0] if i != -1])
    return "No manual data loaded."

# 6. MAIN INTERFACE
if "messages" not in st.session_state: st.session_state.messages = []
col = st.columns([0.15, 0.70, 0.15])[1]
with col:
    st.title("Otimo Aero AI Technician")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.write(msg["content"])

    if query := st.chat_input("Enter system query..."):
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Engine check logic
        if not st.session_state.get("active_engine"):
            if m := re.search(r'(912|914|915|916)', query):
                st.session_state.active_engine = m.group(0)
                st.rerun()
            else:
                st.session_state.messages.append({"role": "assistant", "content": "Please specify engine."})
        else:
            # RAG Execution
            context = search_index(query)
            payload = {"model": "meta-llama/llama-3.1-8b-instruct", "messages": [{"role": "system", "content": f"Context: {context}"}, {"role": "user", "content": query}]}
            response = requests.post(OPENROUTER_URL, headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"}, json=payload).json()["choices"][0]["message"]["content"]
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

# 7. ADMIN
if st.query_params.get("admin") == "true":
    with st.sidebar:
        if st.text_input("Password", type="password") == ADMIN_PASSWORD:
            if files := st.file_uploader("Upload", accept_multiple_files=True):
                # Add parse_and_chunk_pdf logic here
                st.write("Manuals processed.")