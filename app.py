import streamlit as st
from pypdf import PdfReader
import os
import requests
import json
import numpy as np
from openai import OpenAI
import faiss
import time

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================
st.set_page_config(page_title="Otimo Aero AI Desk", page_icon="✈️", layout="wide")

# =====================================================
# 2. API CONFIGURATION
# =====================================================
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
PUSHOVER_USER_KEY = st.secrets.get("PUSHOVER_USER_KEY")
PUSHOVER_APP_TOKEN = st.secrets.get("PUSHOVER_APP_TOKEN")

if not OPENROUTER_API_KEY or not OPENAI_API_KEY:
    st.error("Missing required API credentials.")
    st.stop()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
openai_client = OpenAI(api_key=OPENAI_API_KEY)

INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "faiss_metadata.json"

# =====================================================
# 3. SAFETY & CONTEXT STATE
# =====================================================
COOLDOWN_SECONDS = 5
MAX_QUERY_CHARACTERS = 400
DAILY_TOKEN_BUDGET = 450000

if "documents" not in st.session_state: st.session_state.documents = []
if "last_query_time" not in st.session_state: st.session_state.last_query_time = 0.0
if "daily_token_consumption" not in st.session_state: st.session_state.daily_token_consumption = 0
if "active_engine" not in st.session_state: st.session_state.active_engine = None
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello. Please tell me which engine you are working on: 912UL, 912ULS, 912iS, 914, 915iS or 916iS in order for me to help."}]

# Initialize vector index
if "vector_index" not in st.session_state:
    if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
        try:
            st.session_state.vector_index = faiss.read_index(INDEX_PATH)
            with open(METADATA_PATH, "r", encoding="utf-8") as f: st.session_state.vector_metadata = json.load(f)
        except:
            st.session_state.vector_index = None
            st.session_state.vector_metadata = []
    else:
        st.session_state.vector_index = None
        st.session_state.vector_metadata = []

# =====================================================
# 4. ENGINE & ALERT FUNCTIONS
# =====================================================
def get_embedding(text):
    response = openai_client.embeddings.create(input=[text.replace("\n", " ")], model="text-embedding-3-small")
    return response.data[0].embedding

def send_pushover_alert(title, message):
    if not PUSHOVER_USER_KEY or not PUSHOVER_APP_TOKEN: return
    requests.post("https://api.pushover.net/1/messages.json", data={
        "token": PUSHOVER_APP_TOKEN, "user": PUSHOVER_USER_KEY,
        "title": title, "message": message, "priority": 1, "sound": "siren"
    })

# =====================================================
# 5. DOCUMENT INGESTION
# =====================================================
def rebuild_vector_database(uploaded_files):
    all_chunks = []
    for uploaded_file in uploaded_files:
        reader = PdfReader(uploaded_file)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and len(text.strip()) > 50:
                all_chunks.append({"text": text, "source": uploaded_file.name, "page": page_num + 1})
    
    embeddings = [get_embedding(c["text"]) for c in all_chunks]
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings).astype('float32'))
    
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as f: json.dump(all_chunks, f)
    
    st.session_state.vector_index = index
    st.session_state.vector_metadata = all_chunks
    st.rerun()

# =====================================================
# 6. MAIN CHAT & CONTEXT GATE
# =====================================================
st.title("Otimo Aero")
st.subheader(f"Engine Context: {st.session_state.active_engine if st.session_state.active_engine else 'NOT SET'}")

with st.sidebar:
    st.progress(min(st.session_state.daily_token_consumption / DAILY_TOKEN_BUDGET, 1.0))
    st.caption(f"Tokens: {st.session_state.daily_token_consumption} / {DAILY_TOKEN_BUDGET}")
    uploaded_files = st.file_uploader("Upload Manuals", type=["pdf"], accept_multiple_files=True)
    if uploaded_files: rebuild_vector_database(uploaded_files)

for message in st.session_state.messages:
    with st.chat_message(message["role"]): st.write(message["content"])

user_query = st.chat_input("Enter technical query...")

if user_query:
    # 1. ENGINE CONTEXT GATE
    if st.session_state.active_engine is None:
        if any(x in user_query.upper() for x in ["912", "915", "914"]):
            st.session_state.active_engine = user_query.strip().upper()
            st.rerun()
        else:
            with st.chat_message("assistant"): st.write("### 🔍 Which engine are you enquiring about today?")
            st.stop()

    # 2. BUDGET GUARDRAIL
    if st.session_state.daily_token_consumption >= DAILY_TOKEN_BUDGET:
        send_pushover_alert("🚨 SHUTDOWN", "Daily budget hit.")
        st.error("🚨 EMERGENCY SHUTDOWN: Limit reached.")
        st.stop()

    # 3. LLM SEARCH LOGIC
    with st.chat_message("assistant"):
        with st.spinner("Scanning context..."):
            query_vector = np.array([get_embedding(user_query)]).astype('float32')
            _, indices = st.session_state.vector_index.search(query_vector, 5)
            context = "\n\n".join([st.session_state.vector_metadata[i]["text"] for i in indices[0]])
            
            st.session_state.daily_token_consumption += 9000
            
            final_prompt = f"Supporting technician on {st.session_state.active_engine}. Context: {context}. User asks: {user_query}"
            
            response = requests.post(OPENROUTER_URL, headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"}, 
                                    json={"model": "meta-llama/llama-3.1-8b-instruct", "messages": [{"role": "user", "content": final_prompt}]})
            
            answer = response.json()["choices"][0]["message"]["content"]
            st.write(answer)
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.session_state.messages.append({"role": "assistant", "content": answer})