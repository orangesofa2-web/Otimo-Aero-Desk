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
# 1. PAGE CONFIGURATION & CSS LAYOUT
# =====================================================
st.set_page_config(page_title="Otimo Aero AI Technician", page_icon="✈️", layout="wide")

st.markdown(
    """
    <style>
    div[data-testid="stChatInput"] { max-width: 70% !important; margin: 0 auto !important; }
    .stChatInputContainer { max-width: 70% !important; margin: 0 auto !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# 2. CONFIGURATION & STATE
# =====================================================
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
PUSHOVER_USER_KEY = st.secrets.get("PUSHOVER_USER_KEY")
PUSHOVER_APP_TOKEN = st.secrets.get("PUSHOVER_APP_TOKEN")

if not OPENROUTER_API_KEY or not OPENAI_API_KEY:
    st.error("Missing API credentials.")
    st.stop()

openai_client = OpenAI(api_key=OPENAI_API_KEY)
INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "faiss_metadata.json"

# =====================================================
# 3. CORE LOGIC FUNCTIONS
# =====================================================
def get_embedding(text, model="text-embedding-3-small"):
    return openai_client.embeddings.create(input=[text.replace("\n", " ")], model=model).data[0].embedding

def call_llm(prompt):
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "meta-llama/llama-3.1-8b-instruct",
        "temperature": 0.0,
        "messages": [{"role": "system", "content": "You are a professional aircraft maintenance assistant."}, {"role": "user", "content": prompt}]
    }
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=120)
    return response.json()["choices"][0]["message"]["content"]

# =====================================================
# 4. INITIALIZATION & UI
# =====================================================
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "### 🔧 Welcome to Otimo Aero\n*Disclaimer: This is experimental. Double-check all data against official manuals and consult an iRMT.*"}]

url_params = st.query_params
is_admin_mode = url_params.get("admin") == "true"

def render_workspace():
    st.title("Otimo Aero AI Technician")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.write(message["content"])

if is_admin_mode:
    render_workspace()
else:
    _, col, _ = st.columns([0.15, 0.70, 0.15])
    with col: render_workspace()

# =====================================================
# 5. USER COMMAND PROCESSING
# =====================================================
user_query = st.chat_input("Enter technical maintenance question...")

if user_query:
    # Setup rendering context
    if not is_admin_mode:
        _, col, _ = st.columns([0.15, 0.70, 0.15])
        with col:
            with st.chat_message("user"): st.write(user_query)
    else:
        with st.chat_message("user"): st.write(user_query)

    st.session_state.messages.append({"role": "user", "content": user_query})

    # Prepare Assistant Response
    canvas = st.chat_message("assistant") if is_admin_mode else col.chat_message("assistant")
    with canvas:
        response_placeholder = st.empty()
        
        # SEARCH EXECUTION
        search_query = user_query
        if any(w in user_query.lower() for w in ["test", "troubleshoot", "measure", "gauge"]):
            search_query += " Rotax 9-Series 4-stroke diagnostics 16mm socket"
        
        # (Vector retrieval logic here - truncated for brevity, use previous code for this block)
        context_str = "Manual content..." # In practice, call your FAISS index here
        
        final_prompt = f"""
        You are supporting a licensed technician for Rotax 9-Series 4-stroke engines ONLY.
        
        STRICT RULES:
        1. NO 2-STROKE DATA: Ignore any mention of 503, 582, 618, pre-mix, or points ignition.
        2. TOOL VERIFICATION: 9-Series spark plugs require a 16mm (5/8") socket. If extracts mention 18mm, it is a 2-stroke error—correct it to 16mm.
        3. STRING OVERRIDE: Do not output "Clarification required from user". If data is missing, state "Manual data gaps present".
        
        ---
        MANUAL EXTRACTS: {context_str}
        USER QUESTION: {user_query}
        """
        
        response = call_llm(final_prompt)
        response_placeholder.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})