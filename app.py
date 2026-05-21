import os
import re
import json
import numpy as np
import requests
import faiss
import time
import streamlit as st
from pypdf import PdfReader
from openai import OpenAI

# =====================================================
# 1. PAGE CONFIGURATION & CSS
# =====================================================
st.set_page_config(page_title="Otimo Aero AI Technician", page_icon="✈️", layout="wide")

st.markdown("""
    <style>
    div[data-testid="stChatInput"] { max-width: 70% !important; margin: 0 auto !important; }
    .stChatInputContainer { max-width: 70% !important; margin: 0 auto !important; }
    .block-container { padding-bottom: 150px !important; } 
    </style>
    """, unsafe_allow_html=True)

# =====================================================
# 2. CONFIGURATION & REGISTRY (TRIMMED FOR STABILITY)
# =====================================================
# (Keep your existing SPEC_REGISTRY here, ensure NO '>' symbols exist in any string)

# =====================================================
# 3. CORE LOGIC
# =====================================================
if "active_engine" not in st.session_state: st.session_state.active_engine = None
if "messages" not in st.session_state: 
    st.session_state.messages = [{
        "role": "assistant", 
        "content": "### 🔧 Engine Selection Required\nWelcome to the workbench! Before we look up any technical maintenance details, we need to lock onto your precise engine configuration.\n\n**🚨 IMPORTANT MAINTENANCE DIRECTIVE / TECHNICAL DISCLAIMER 🚨**\nThis AI system is highly experimental and serves strictly as an informational guide. All users must cross-reference and double-check instructions, tolerances, and part arrays against official hardcopy documentation before altering any flight system.\n\n**Please reply with the specific engine type:**\n* **912UL** | **912ULS** | **912iS** | **914** | **915iS** | **916iS**"
    }]

_, center_console, _ = st.columns([0.15, 0.70, 0.15])

with center_console:
    st.title("Otimo Aero AI Technician")
    if st.session_state.active_engine:
        st.markdown(f"#### 🛠️ Workspace Status\n**Engine:** `{st.session_state.active_engine}`")
        st.divider()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.write(msg["content"])

user_query = st.chat_input("Enter your request...")

if user_query:
    if st.session_state.active_engine is None:
        match = re.search(r'(912\s*uls|915|916)', user_query.lower())
        if match:
            st.session_state.active_engine = "915IS" if "915" in match.group(0) else match.group(0).upper().replace(" ", "")
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.rerun()
        else:
            st.error("Please specify a valid engine profile.")
    else:
        # Process query directly without aggressive full-page refreshes
        st.session_state.messages.append({"role": "user", "content": user_query})
        with center_console.chat_message("user"):
            st.write(user_query)
        
        # Call LLM and append to session state
        # ... (Your existing RAG/Call LLM logic here)