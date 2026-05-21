import os
import re
import json
import numpy as np
import requests
import faiss
import streamlit as st
from pypdf import PdfReader
from openai import OpenAI

# =====================================================
# 1. PAGE CONFIGURATION & CSS
# =====================================================
st.set_page_config(page_title="Otimo Aero AI Technician", page_icon="✈️", layout="wide")
st.markdown("""<style>
    div[data-testid="stChatInput"] { max-width: 70% !important; margin: 0 auto !important; }
    .stChatInputContainer { max-width: 70% !important; margin: 0 auto !important; }
    .block-container { padding-bottom: 150px !important; } 
    </style>""", unsafe_allow_html=True)

# =====================================================
# 2. VECTOR ENGINE (THE "BRAINS")
# =====================================================
INDEX_PATH, METADATA_PATH = "faiss_index.bin", "faiss_metadata.json"
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def get_embedding(text):
    return openai_client.embeddings.create(input=[text.replace("\n", " ")], model="text-embedding-3-small").data[0].embedding

def rebuild_vector_database(uploaded_files):
    all_chunks = []
    for uploaded_file in uploaded_files:
        reader = PdfReader(uploaded_file)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                words = re.sub(r'\s+', ' ', text).split()
                for i in range(0, len(words), 100):
                    chunk = " ".join(words[i:i+100])
                    if len(chunk) > 50: all_chunks.append({"text": chunk, "source": uploaded_file.name})
    
    embeddings = [get_embedding(c["text"]) for c in all_chunks]
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings).astype('float32'))
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "w") as f: json.dump(all_chunks, f)
    st.rerun()

# Load Index
if "vector_index" not in st.session_state:
    if os.path.exists(INDEX_PATH):
        st.session_state.vector_index = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH, "r") as f: st.session_state.vector_metadata = json.load(f)
    else: st.session_state.vector_index = None

# =====================================================
# 3. MAIN INTERFACE
# =====================================================
if "engine" not in st.session_state: st.session_state.engine = None
if "messages" not in st.session_state: st.session_state.messages = []

st.title("Otimo Aero AI Technician")
if st.session_state.engine:
    st.markdown(f"#### 🛠️ Workspace Status\n**Engine:** `{st.session_state.engine}`")
    st.divider()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.write(msg["content"])

query = st.chat_input("Enter Engine or Question...")

if query:
    with st.chat_message("user"): st.write(query)
    st.session_state.messages.append({"role": "user", "content": query})

    if st.session_state.engine is None:
        match = re.search(r'(912\s*uls|915|916)', query.lower())
        if match:
            st.session_state.engine = match.group(0).upper()
            st.rerun()
        else:
            st.session_state.messages.append({"role": "assistant", "content": "🚨 Specify Engine: 912ULS, 915iS, 916iS"})
            st.rerun()
    else:
        # Retrieve context from FAISS
        context_str = "No manual found."
        if st.session_state.vector_index:
            q_vec = np.array([get_embedding(query)]).astype('float32')
            _, idxs = st.session_state.vector_index.search(q_vec, 2)
            context_str = "\n".join([st.session_state.vector_metadata[i]['text'] for i in idxs[0]])

        # LLM Call
        response = call_llm(context_str, query) # (Insert LLM Call logic here)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()