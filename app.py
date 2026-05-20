import streamlit as st
from pypdf import PdfReader
import os
import re
import requests
import json
import numpy as np
from openai import OpenAI
import faiss
import time

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Otimo Aero AI Desk",
    page_icon="✈️",
    layout="wide"
)

# =====================================================
# 2. DUAL API CONFIGURATION SAFETY GATES
# =====================================================
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
PUSHOVER_USER_KEY = st.secrets.get("PUSHOVER_USER_KEY")
PUSHOVER_APP_TOKEN = st.secrets.get("PUSHOVER_APP_TOKEN")

if not OPENROUTER_API_KEY or not OPENAI_API_KEY:
    st.error("Missing required API credentials in Streamlit Secrets. Ensure both OPENROUTER_API_KEY and OPENAI_API_KEY are configured.")
    st.stop()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Initialize OpenAI Client strictly for high-dimensional semantic embeddings
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Vector File Paths for Persistent Storage on the Server
INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "faiss_metadata.json"

# =====================================================
# 3. SAFETY GUARDRAIL PARAMETERS (UPDATED BENCHMARKS)
# =====================================================
COOLDOWN_SECONDS = 5        # Minimum wait time between consecutive submissions
MAX_QUERY_CHARACTERS = 400  # Max size allowed for a single question
DAILY_TOKEN_BUDGET = 450000 # Emergency circuit breaker for exactly 50 lookups a day

# =====================================================
# 4. SESSION STATE INITIALIZATION
# =====================================================
if "documents" not in st.session_state:
    st.session_state.documents = []

# Rate Limiting & Alert State Trackers
if "last_query_time" not in st.session_state:
    st.session_state.last_query_time = 0.0

if "daily_token_consumption" not in st.session_state:
    st.session_state.daily_token_consumption = 0

if "alert_triggered_today" not in st.session_state:
    st.session_state.alert_triggered_today = False

if "active_engine" not in st.session_state:
    st.session_state.active_engine = None

# PERSISTENT TOPIC TRACKER: Keeps the maintenance task locked in across turns
if "active_topic" not in st.session_state:
    st.session_state.active_topic = None

# Load local vector index into live container memory on startup if it exists
if "vector_index" not in st.session_state:
    if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
        try:
            st.session_state.vector_index = faiss.read_index(INDEX_PATH)
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                st.session_state.vector_metadata = json.load(f)
            st.session_state.documents = list(set(m["source"] for m in st.session_state.vector_metadata))
        except Exception:
            st.session_state.vector_index = None
            st.session_state.vector_metadata = []
    else:
        st.session_state.vector_index = None
        st.session_state.vector_metadata = []

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello. True semantic vector production engine active. Engine context must be specified to initialize workspace panels."
        }
    ]

if "pending_clarification" not in st.session_state:
    st.session_state.pending_clarification = None

# =====================================================
# 5. TECHNICAL SAFETY LAYERS & CORE ENGINES
# =====================================================
def requires_variant(query: str) -> bool:
    q = query.lower().replace(" ", "").replace("-", "")
    return "912" in q and not any(v in q for v in ["uls", "ul", "is"])

def invalid_configuration(query: str) -> bool:
    q = query.lower().replace(" ", "").replace("-", "")
    carb_terms = ["carb", "sync", "balance", "float", "choke"]
    injected_engines = ["915", "916", "912is"]
    
    is_carb_query = any(t in q for t in carb_terms)
    is_injected = any(e in q for e in injected_engines)
    return is_carb_query and is_injected

# Helper: Request Dense Vector Coordinates from OpenAI Embeddings Engine
def get_embedding(text: str, model="text-embedding-3-small"):
    cleaned_text = text.replace("\n", " ")
    response = openai_client.embeddings.create(input=[cleaned_text], model=model)
    return response.data[0].embedding

# Helper: Professional Grade Pushover Notification Delivery Engine
def send_pushover_alert(title: str, message: str):
    if not PUSHOVER_USER_KEY or not PUSHOVER_APP_TOKEN:
        return
    try:
        url = "https://api.pushover.net/1/messages.json"
        payload = {
            "token": PUSHOVER_APP_TOKEN,
            "user": PUSHOVER_USER_KEY,
            "title": title,
            "message": message,
            "priority": 1,
            "sound": "siren"
        }
        requests.post(url, data=payload, timeout=10)
    except Exception:
        pass

# =====================================================
# 6. DOCUMENT INGESTION & VECTOR MATRIX BUILDER
# =====================================================
def rebuild_vector_database(uploaded_files):
    all_chunks = []
    
    for uploaded_file in uploaded_files:
        try:
            reader = PdfReader(uploaded_file)
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text and len(page_text.strip()) > 50:
                    all_chunks.append({
                        "text": page_text,
                        "source": uploaded_file.name,
                        "page": page_num + 1
                    })
        except Exception as e:
            st.error(f"Error parsing {uploaded_file.name}: {str(e)}")
            
    if all_chunks:
        embeddings_list = []
        metadata_list = []
        progress_bar = st.progress(0)
        st.write(f"Vectorising {len(all_chunks)} manual pages via OpenAI API...")
        
        for idx, chunk in enumerate(all_chunks):
            try:
                vec = get_embedding(chunk["text"])
                embeddings_list.append(vec)
                metadata_list.append(chunk)
            except Exception as api_err:
                st.error(f"Embedding failure on page unit {idx}: {str(api_err)}")
            progress_bar.progress((idx + 1) / len(all_chunks))
            
        if embeddings_list:
            dimension = len(embeddings_list[0])
            np_embeddings = np.array(embeddings_list).astype('float32')
            
            index = faiss.IndexFlatL2(dimension)
            index.add(np_embeddings)
            
            faiss.write_index(index, INDEX_PATH)
            with open(METADATA_PATH, "w", encoding="utf-8") as f:
                json.dump(metadata_list, f, ensure_ascii=False, indent=2)
                
            st.session_state.vector_index = index
            st.session_state.vector_metadata = metadata_list
            st.session_state.documents = list(set(m["source"] for m in metadata_list))
            st.success("Universal semantic database built and stored successfully!")
            st.rerun()

# =====================================================
# 7. OPENROUTER PRODUCTION HANDSHAKE (LLAMA 3.1 8B)
# =====================================================
def call_llm(prompt: str):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "meta-llama/llama-3.1-8b-instruct",
        "temperature": 0.0,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are the lead technical AI desk assistant for Otimo Aero, providing maintenance support directly to technicians working on aircraft. "
                    "You output answers in a strict, professional, itemized layout. No conversational fluff, meta-references, or unhelpful remarks."
                )
            },
            {"role": "user", "content": prompt}
        ],
        "providers": {
            "order": ["Lepton", "Together"],
            "allow_fallbacks": True
        }
    }
    
    response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
    if response.status_code != 200:
        raise Exception(response.text)
    return response.json()["choices"][0]["message"]["content"]

# =====================================================
# 8. SIDEBAR CONTROL PANEL (ADMIN GATE LOCKED)
# =====================================================
url_params = st.query_params

if url_params.get("admin") == "true":
    with st.sidebar:
        st.header("⚙️ Admin Control Panel")
        
        if not st.session_state.documents:
            uploaded_files = st.file_uploader(
                "Upload Technical Manuals",
                type=["pdf"],
                accept_multiple_files=True
            )
            if uploaded_files:
                with st.spinner("Executing high-dimensional conceptual indexing..."):
                    rebuild_vector_database(uploaded_files)
        else:
            st.success("🔒 System Manuals Locked into Memory")
            if st.button("Clear & Re-upload Manuals"):
                if os.path.exists(INDEX_PATH): os.remove(INDEX_PATH)
                if os.path.exists(METADATA_PATH): os.remove(METADATA_PATH)
                st.session_state.vector_index = None
                st.session_state.vector_metadata = []
                st.session_state.documents = []
                st.session_state.active_topic = None
                st.session_state.rerun()

        st.divider()
        st.metric("Indexed Manuals", len(st.session_state.documents))
        st.metric("Searchable Vector Units", len(st.session_state.vector_metadata) if st.session_state.vector_metadata else 0)
        
        st.divider()
        st.subheader("Guardrail Budget Tracking")
        st.progress(min(st.session_state.daily_token_consumption / DAILY_TOKEN_BUDGET, 1.0))
        st.caption(f"Daily Token Counter: {st.session_state.daily_token_consumption} / {DAILY_TOKEN_BUDGET}")

# =====================================================
# 9. MAIN CHAT DISPLAY
# =====================================================
st.title("Otimo Aero")

status_line = f"Workspace Status — Engine Profile: {st.session_state.active_engine if st.session_state.active_engine else 'Please input your engine model; 912U, 912ULS, 912iS, 914, 915iS or 916iS'}"
if st.session_state.active_topic:
    status_line += f" | Current Maintenance Task: {st.session_state.active_topic}"
st.subheader(status_line)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# =====================================================
# 10. USER COMMAND RUNNER WITH ARCHITECTURE HOOKS
# =====================================================
user_query = st.chat_input("Enter technical maintenance question...")

if user_query:
    current_time = time.time()
    time_passed = current_time - st.session_state.last_query_time
    
    with st.chat_message("user"):
        st.write(user_query)

    # ENGINE CONTEXT GATE
    if st.session_state.active_engine is None:
        engine_match = re.search(r'(912\s*uls|912\s*ul|912\s*is|914|915\s*is|915|916\s*is|916)', user_query.lower())
        if engine_match:
            normalized_engine = engine_match.group(1).upper().replace(" ", "")
            st.session_state.active_engine = normalized_engine
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"### 🔓 CONTEXT ACTIVATED\nEngine profile set to **ROTAX {normalized_engine}**. Core workbench interface channels are ready. Please enter your primary technical maintenance query."
            })
            st.rerun()
        else:
            assistant_prompt = "### 🔍 ENGINE CONTEXT REQUIRED\nPlease explicitly declare the targeted engine variant model platform (**912 UL**, **912 ULS**, **912 iS**, **914**, **915 iS**, or **916 iS**) to establish session memory parameters."
            with st.chat_message("assistant"):
                st.write(assistant_prompt)
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.session_state.messages.append({"role": "assistant", "content": assistant_prompt})
            st.stop()

    # TOPIC EXTRACTOR LOGIC: Detects if the user is changing the core procedure topic
    change_topic_match = re.search(r'(purge|oil|plug|spark|gap|torque|carb|balance|sync)', user_query.lower())
    if change_topic_match and "tool" not in user_query.lower():
        # Set a clean text anchor based on keyword detection
        if "purge" in user_query.lower() or "oil" in user_query.lower():
            st.session_state.active_topic = "OIL PURGING"
        elif "plug" in user_query.lower() or "gap" in user_query.lower():
            st.session_state.active_topic = "SPARK PLUG INSPECTION"
        elif "carb" in user_query.lower() or "sync" in user_query.lower() or "balance" in user_query.lower():
            st.session_state.active_topic = "CARBURETOR SYNCHRONIZATION"

    # TEST TRIGGER
    if user_query.strip() == "TEST_ALERT_NOW":
        st.session_state.daily_token_consumption = DAILY_TOKEN_BUDGET + 1000

    # GUARDRAIL LAYER A: Cooldown Timer Enforcement
    if time_passed < COOLDOWN_SECONDS:
        wait_remainder = int(COOLDOWN_SECONDS - time_passed)
        error_msg = f"⏳ **RATE LIMIT TRIGGERED:** Please wait {wait_remainder} more seconds before submitting another question to protect system stability."
        with st.chat_message("assistant"):
            st.warning(error_msg)
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        st.stop()

    # GUARDRAIL LAYER B: Query Size Hard Cap
    if len(user_query) > MAX_QUERY_CHARACTERS:
        error_msg = f"⚠️ **INPUT OVERFLOW:** Your entry is too long ({len(user_query)} characters). Questions are limited to {MAX_QUERY_CHARACTERS} characters."
        with st.chat_message("assistant"):
            st.error(error_msg)
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        st.stop()

    # GUARDRAIL LAYER C: Daily Token Budget Safety Brake
    if st.session_state.daily_token_consumption >= DAILY_TOKEN_BUDGET:
        if not st.session_state.alert_triggered_today:
            send_pushover_alert(
                title="🚨 Otimo Aero: Daily Budget Spent",
                message=f"The application has successfully hit its safety cap limit of {DAILY_TOKEN_BUDGET} tokens. Interface API traffic locked."
            )
            st.session_state.alert_triggered_today = True

        error_msg = "🚨 **EMERGENCY SHUTDOWN:** The application has reached its maximum daily data allotment. API requests have been locked down to prevent balance exhaustion. Please check again tomorrow."
        with st.chat_message("assistant"):
            st.error(error_msg)
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.session_state.messages.append({"role