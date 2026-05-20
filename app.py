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
                st.error(f"Embedding failure on page unit