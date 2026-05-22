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

# 5. DYNAMIC MASTER SPEC REGISTRY & STATE
SPEC_REGISTRY = {
    "OIL CHANGE / MAGNETIC PLUG INSPECTION": {
        "reasoning_points": [
            "Draining oil when the engine is WARM/HOT is mandatory to ensure wear particles are suspended and drain out completely (scavenging). Cold oil is thick and will not drain fully.",
            "Torque wrenches are precision calibration instruments for TIGHTENING ONLY. Using them for loosening permanently damages their internal mechanism.",
            "The Crankcase Magnetic Plug uses a precision TAPERED seat for a metal-to-metal seal. Adding a washer or sealant will prevent a proper seal.",
            "The Oil Tank Drain Screw (Sump) uses a soft copper sealing ring that is crushed on tightening. It is a one-time-use item and MUST be replaced.",
            "Lubricating threads and gaskets with clean engine oil prevents thread galling (cold welding)."
        ],
        "specs_and_tooling_markdown": """
- **Engine Pre-Condition:** Drain the oil only when **WARM or HOT**.
- **Oil Tank Drain Screw (Sump Plug):** Torque: **25 Nm (221 in. lb)**. MUST use a NEW copper ring.
- **Crankcase Magnetic Plug (Tapered Seal):** Torque: Strictly **20 Nm (177 in. lb)**. DO NOT use washers or sealant.
- **Oil Filter:** Hand-tighten until contact, then tighten a further **3/4 turn**. DO NOT use a torque wrench.
"""
    },
    "CARBURETOR SYNCHRONIZATION": {
        "reasoning_points": [
            "Mechanical synchronization (adjusting cable slack) on a COLD engine is a mandatory prerequisite.",
            "The idle RPM synchronization is the ONLY phase where a direct adjustment is made (20 mbar tolerance).",
            "The cruise power (3500-4000 RPM) check is a VERIFICATION-ONLY step with a zero-tolerance (0 mbar) requirement.",
            "Pressure deviation at cruise power signifies a serious mechanical fault."
        ],
        "specs_and_tooling_markdown": """
- **Step 1: Mechanical Synchronization (COLD ENGINE):** Cables must have minimum free play of 1 mm (0.04 in).
- **Step 2 (Part A): Idle Adjustment (1800-2000 RPM):** Max pressure difference is **20 mbar (0.29 psi)**.
- **Step 2 (Part B): Cruise Verification (3500-4000 RPM):** Required difference is **0 mbar**. DO NOT adjust here.
"""
    }
}

if "active_engine" not in st.session_state: st.session_state.active_engine = None
if "active_topic" not in st.session_state: st.session_state.active_topic = None
if "messages" not in st.session_state: 
    st.session_state.messages = [{
        "role": "assistant", 
        "content": "### 🔧 Engine Selection Required\nWelcome to the workbench! Please reply with the specific engine type you are working on today:\n* **912UL** | **912ULS** | **912iS** | **914** | **915iS** | **916iS**"
    }]
if "embed_cache" not in st.session_state: st.session_state.embed_cache = json.load(open(CACHE_PATH, "r")) if os.path.exists(CACHE_PATH) else {}
if "vector_index" not in st.session_state:
    if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
        st.session_state.vector_index = faiss.read_index(INDEX_PATH)
        st.session_state.vector_metadata = json.load(open(METADATA_PATH, "r"))
    else: st.session_state.vector_index, st.session_state.vector_metadata = None, []

# 6. VECTOR FUNCTIONS & LLM HUB
def get_embeddings_batched(texts, model="text-embedding-3-small"):
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

def search_index(query):
    if st.session_state.vector_index is not None:
        q_vec = np.array(get_embeddings_batched([query])).astype('float32')
        faiss.normalize_L2(q_vec)
        distances, idxs = st.session_state.vector_index.search(q_vec, 3)
        chunks = []
        for score, idx in zip(distances[0], idxs[0]):
            if idx != -1 and score > 0.55 and idx < len(st.session_state.vector_metadata):
                chunks.append(st.session_state.vector_metadata[idx]['text'])
        return "\n\n---\n\n".join(chunks) if chunks else "No manual data extracted."
    return "Vector database not initialized."

def call_llm(user_context, chat_history):
    system_prompt = "You are 'Otimo Inspector', an expert AI mentor for ROTAX engines. Address the issue using a 3-part format: 1. THE WORKBENCH PROCEDURE, 2. INSPECTOR'S SAFETY BRIEF, 3. REQUIRED SPECS & TOOLING."
    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history[-4:]: messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_context})
    payload = {"model": "meta-llama/llama-3.1-8b-instruct", "temperature": 0.1, "messages": messages}
    return requests.post(OPENROUTER_URL, headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"}, json=payload).json()["choices"][0]["message"]["content"]

def parse_and_chunk_pdf(uploaded_files):
    all_chunks = []
    for f in uploaded_files:
        reader = PdfReader(f)
        for i, page in enumerate(reader.pages):
            if text := page.extract_text(): all_chunks.append({"text": re.sub(r'\s+', ' ', text).strip(), "source": f.name, "page": i + 1})
    if all_chunks:
        with st.spinner("Processing optimization tokens into secure cache layer..."):
            embeddings = np.array(get_embeddings_batched([c["text"] for c in all_chunks])).astype('float32')
            faiss.normalize_L2(embeddings)
            index = faiss.IndexFlatIP(len(embeddings[0])); index.add(embeddings)
            faiss.write_index(index, INDEX_PATH)
            json.dump(all_chunks, open(METADATA_PATH, "w"))
            st.success("Universal localized system vector database synchronized!")
            st.rerun()

# 7. MAIN UI & EXECUTION
col_layout = st.columns([0.15, 0.70, 0.15])[1]
with col_layout:
    st.title("Otimo Aero AI Technician")
    if st.session_state.active_engine:
        st.markdown(f"#### 🛠️ Workspace Connected \n**Engine:** `ROTAX {st.session_state.active_engine}`")
        st.divider()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.write(msg["content"])
    
    if user_query := st.chat_input("Enter engine profile code or technical question..."):
        with st.chat_message("user"): st.write(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        if st.session_state.active_engine is None:
            if match := re.search(r'(912\s*uls|912\s*ul|912\s*is|914|915\s*is|916\s*is|916)', user_query.lower()):
                st.session_state.active_engine = match.group(1).upper().replace(" ", "")
                st.session_state.messages.append({"role": "assistant", "content": f"### 🔓 WORKSPACE UNLOCKED\nEngine profile securely set to **ROTAX {st.session_state.active_engine}**."})
                st.rerun()
            else:
                st.session_state.messages.append({"role": "assistant", "content": "⚠️ **ENGINE PROFILE UNINITIALISED**\nState exact specification key setup to unlock workbench access: **912UL | 912ULS | 912iS | 914 | 915iS | 916iS**"})
                st.rerun()
        else:
            with st.spinner("Scanning vectorized indices..."):
                context_str = search_index(f"ROTAX {st.session_state.active_engine} {user_query}")
                
                # Setup Topic Data
                topic_data = SPEC_REGISTRY.get("OIL CHANGE / MAGNETIC PLUG INSPECTION") if "oil" in user_query.lower() else None
                reasoning = "\n".join([f"- {p}" for p in topic_data["reasoning_points"]]) if topic_data else "Verify maintenance alignment."
                specs = topic_data["specs_and_tooling_markdown"] if topic_data else "No lookup values configured."

                full_context = f"REASONING:\n{reasoning}\n\nSPECS:\n{specs}\n\nENGINE: ROTAX {st.session_state.active_engine}\nMANUAL EXTRACTS:\n{context_str}"
                
                response = call_llm(full_context, st.session_state.messages)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

# 8. ADMIN ACCESS
if st.query_params.get("admin") == "true":
    st.markdown("<style>[data-testid='stSidebar'] { display: block !important; }</style>", unsafe_allow_html=True)
    with st.sidebar:
        st.header("🔑 Administrative Access")
        if st.text_input("Password", type="password") == ADMIN_PASSWORD:
            if files := st.file_uploader("Upload Manuals", accept_multiple_files=True): parse_and_chunk_pdf(files)