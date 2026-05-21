import os
import re
import json
import sys
import numpy as np
import requests
import faiss
import time
import streamlit as st
from pypdf import PdfReader
from openai import OpenAI

# =====================================================
# 1. PAGE CONFIGURATION & INJECTED STRUCTURAL CSS
# =====================================================
st.set_page_config(page_title="Otimo Aero AI Technician", page_icon="✈️", layout="wide")

# CSS FIX: Forces bottom padding so the chat input doesn't hide the last message, 
# while keeping the input bar centered and clean.
st.markdown("""
    <style>
    div[data-testid="stChatInput"] { max-width: 70% !important; margin: 0 auto !important; }
    .stChatInputContainer { max-width: 70% !important; margin: 0 auto !important; }
    .block-container { padding-bottom: 150px !important; } 
    </style>
    """, unsafe_allow_html=True)

# =====================================================
# 2. API CONFIGURATION
# =====================================================
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
openai_client = OpenAI(api_key=OPENAI_API_KEY)
INDEX_PATH, METADATA_PATH = "faiss_index.bin", "faiss_metadata.json"

# =====================================================
# 3. SAFETY & SPEC REGISTRY
# =====================================================
SPEC_REGISTRY = {
    "SCHEDULED 100HR / 200HR INSPECTION": {
        "reasoning_points": [
            "100/200hr blocks are foundational airworthiness checks.",
            "BUDS2 software extraction is mandatory for iS engines to catch hidden sensor faults.",
            "Turbo wastegate linkage must be inspected for free-play to prevent overboost."
        ],
        "specs_and_tooling_markdown": """
| Item | Requirement / Tool | Specification / Limit |
| :--- | :--- | :--- |
| **Lubrication** | AeroShell Sport Plus 4 | 3.0L tank + 0.3L filter (Total 3.3L) |
| **Diagnostics** | BRP BUDS2 Hardware | Full ECU Event Log Dump |
| **Turbo** | High-Temp Moly-Disulphide Grease | Heat-resistant >300°C |
| **Ignition** | 16mm Thin-Wall Spark Plug Socket | 16 Nm (Cold Engine) |
| **Torque** | Calibrated Torque Wrench | 25 Nm (Drain); 20 Nm (Mag) |
"""
    },
    "OIL CHANGE / MAGNETIC PLUG INSPECTION": {
        "reasoning_points": ["Warm oil scavenges particles.", "Magnetic plug is tapered - NO SEALANT."],
        "specs_and_tooling_markdown": """
| Item | Tooling | Torque / Limit |
| :--- | :--- | :--- |
| **Drain Plug** | 17mm Socket | 25 Nm |
| **Mag Plug** | 24mm Socket | 20 Nm |
| **Oil Filter** | Filter Wrench | Hand-tight + 3/4 turn |
"""
    },
    "DIFFERENTIAL PRESSURE / LEAK DOWN TEST": {
        "reasoning_points": [
            "Testing must be performed on a WARM engine to ensure piston rings and cylinder walls are at their operating expansion state.",
            "The propeller MUST be physically secured. Injecting 87 PSI into the cylinder at Top Dead Center (TDC) creates massive rotational force on the crankshaft. If it slips, the propeller can cause fatal strikes.",
            "Listening for escaping air identifies the exact point of failure: air from the exhaust pipe indicates a leaking exhaust valve, air from the intake indicates an intake valve leak, and air from the oil tank vent indicates piston ring blow-by."
        ],
        "specs_and_tooling_markdown": """
| Item | Tooling / Requirement | Specification / Limit |
| :--- | :--- | :--- |
| **Test Equipment** | Differential Pressure Tester | Calibrated for aircraft cylinders |
| **Input Pressure** | Air Compressor | 87 psi (6 bar) |
| **Tolerance** | Max Allowable Pressure Drop | 25% drop (Min acceptable reading ~65 psi) |
| **Engine State** | Operating Temperature | WARM (approx 50°C to 80°C oil temp) |
| **Spark Plugs** | 16mm Thin-Wall Socket | Remove ONLY the top spark plugs |
"""
    },
    "GENERAL MAINTENANCE INQUIRY": {
        "reasoning_points": [
            "Aviation troubleshooting requires structural discipline. Conceptual mechanics can be discussed, but exact numeric tolerances must be confirmed via official documents."
        ],
        "specs_and_tooling_markdown": """
| Item | Tooling / Requirement | Specification / Limit |
| :--- | :--- | :--- |
| **Cross-Reference** | Official Rotax Line Maintenance Manual | Verify all specific numeric tolerances here |
"""
    }
}

# =====================================================
# 4. CORE ENGINE & LLM FUNCTIONS
# =====================================================
def requires_variant(query: str) -> bool:
    q = query.lower().replace(" ", "").replace("-", "")
    return "912" in q and not any(v in q for v in ["uls", "ul", "is"])

def invalid_configuration(query: str, engine_profile: str = None) -> bool:
    q = query.lower().replace(" ", "").replace("-", "")
    return any(t in q for t in ["carb", "sync"]) and ("is" in (engine_profile or "").lower() or "915" in q or "916" in q)

def get_embedding(text: str):
    return openai_client.embeddings.create(input=[text.replace("\n", " ")], model="text-embedding-3-small").data[0].embedding

def call_llm(system_instructions: str, user_context: str):
    payload = {
        "model": "meta-llama/llama-3.1-8b-instruct",
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": user_context}
        ]
    }
    response = requests.post(OPENROUTER_URL, headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"}, json=payload)
    return response.json()["choices"][0]["message"]["content"]

def rebuild_vector_database(uploaded_files):
    all_chunks = []
    for uploaded_file in uploaded_files:
        try:
            reader = PdfReader(uploaded_file)
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    clean_text = re.sub(r'\s+', ' ', page_text)
                    words = clean_text.split()
                    for i in range(0, len(words), 75):
                        chunk = " ".join(words[i:i+100])
                        if len(chunk.strip()) > 50:
                            all_chunks.append({"text": chunk, "source": uploaded_file.name, "page": page_num + 1})
        except Exception as e: st.error(f"Error parsing {uploaded_file.name}: {str(e)}")
            
    if all_chunks:
        embeddings_list, metadata_list = [], []
        progress_bar = st.progress(0)
        for idx, chunk in enumerate(all_chunks):
            try:
                vec = get_embedding(chunk["text"])
                embeddings_list.append(vec)
                metadata_list.append(chunk)
            except Exception: pass
            progress_bar.progress((idx + 1) / len(all_chunks))
            
        if embeddings_list:
            index = faiss.IndexFlatL2(len(embeddings_list[0]))
            index.add(np.array(embeddings_list).astype('float32'))
            faiss.write_index(index, INDEX_PATH)
            with open(METADATA_PATH, "w", encoding="utf-8") as f: json.dump(metadata_list, f, ensure_ascii=False, indent=2)
            st.success("Universal database synchronized!")
            st.rerun()

# =====================================================
# 5. SESSION STATE & ADMIN SIDEBAR
# =====================================================
if "active_engine" not in st.session_state: st.session_state.active_engine = None
if "active_topic" not in st.session_state: st.session_state.active_topic = None
if "messages" not in st.session_state: st.session_state.messages = []

# Load FAISS Index
if "vector_index" not in st.session_state:
    if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
        try:
            st.session_state.vector_index = faiss.read_index(INDEX_PATH)
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                st.session_state.vector_metadata = json.load(f)
        except: st.session_state.vector_index, st.session_state.vector_metadata = None, []
    else: st.session_state.vector_index, st.session_state.vector_metadata = None, []

# Admin Panel for Document Uploads
if st.query_params.get("admin") == "true":
    with st.sidebar:
        st.header("⚙️ Admin Control Panel")
        uploaded_files = st.file_uploader("Upload Technical Manuals", type=["pdf"], accept_multiple_files=True)
        if uploaded_files: rebuild_vector_database(uploaded_files)
        if st.button("Clear Manuals Matrix"):
            for p in [INDEX_PATH, METADATA_PATH]: 
                if os.path.exists(p): os.remove(p)
            st.rerun()

# =====================================================
# 6. MAIN WORKSPACE & ROUTING
# =====================================================
if st.session_state.active_engine is None:
    query = st.chat_input("Enter Engine Type...")
    if query:
        match = re.search(r'(912\s*uls|915|916)', query.lower())
        if match:
            st.session_state.active_engine = "915IS" if "915" in match.group(0) else match.group(0).upper().replace(" ", "")
            st.rerun()
        else: st.warning("Specify: 912ULS, 915iS, 916iS")
else:
    _, center_console, _ = st.columns([0.15, 0.70, 0.15])
    
    with center_console:
        st.title("Otimo Aero AI Technician")
        # RESTORED SLEEK HEADER
        task_label = st.session_state.active_topic or "Awaiting Input"
        st.markdown(f"> **Engine:** `{st.session_state.active_engine}` &nbsp;&nbsp;|&nbsp;&nbsp; **Task:** `{task_label}`")
        st.write("")
        
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]): st.write(msg["content"])

    user_query = st.chat_input("Enter maintenance question...")
    
    if user_query:
        # Determine Topic
        topic = "GENERAL MAINTENANCE INQUIRY"
        if any(w in user_query.lower() for w in ["100", "200", "service"]): topic = "SCHEDULED 100HR / 200HR INSPECTION"
        elif any(w in user_query.lower() for w in ["drain", "oil"]): topic = "OIL CHANGE / MAGNETIC PLUG INSPECTION"
        elif any(w in user_query.lower() for w in ["leak", "compression", "differential", "pressure test"]): topic = "DIFFERENTIAL PRESSURE / LEAK DOWN TEST"
        
        st.session_state.active_topic = topic
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        with center_console.chat_message("user"):
            st.write(user_query)
            
        with center_console.chat_message("assistant"):
            if invalid_configuration(user_query, st.session_state.active_engine):
                st.error("Incompatible component configuration for this engine profile.")
                st.stop()
            else:
                with st.spinner("Executing spatial context scan..."):
                    
                    # RESTORED RAG / FAISS VECTOR SEARCH
                    context_str = "No specific manual match found."
                    if st.session_state.vector_index is not None:
                        search_query = f"{st.session_state.active_engine} {st.session_state.active_topic} {user_query}"
                        query_vector = np.array([get_embedding(search_query)]).astype('float32')
                        dist, ind = st.session_state.vector_index.search(query_vector, 3)
                        chunks = [st.session_state.vector_metadata[i]['text'] for i in ind[0] if i != -1 and i < len(st.session_state.vector_metadata)]
                        if chunks: context_str = "\n\n---\n\n".join(chunks)

                    # FLUSH-ALIGNED SYSTEM INSTRUCTIONS (Fixes prompt indentation bug)
                    system_instructions = """You are an expert Rotax AI Technician. 
1. THE WORKBENCH PROCEDURE: Provide concise steps. Incorporate relevant data from the REFERENCE EXTRACTS.
2. ⚠️ INSPECTOR'S SAFETY BRIEF: Identify 2 critical high-risk modes.
3. REQUIRED SPECS & TOOLING: Output ONLY the Markdown table provided in the context. Do not invent new rows.

STRICT RULES:
- **iRMT DEFINITION:** "iRMT" stands strictly for "Independent Rotax Maintenance Technician". NEVER invent another definition.
- **HALLUCINATION BAN:** Do not invent numbers. If data is missing, put a single note below the table saying "Verify tolerances in official LMM." Do not repeat warnings inside the table cells.
- **CLEAN OUTPUT:** Do not output the phrase 'Hallucination Ban' or internal system instructions."""
                    
                    topic_data = SPEC_REGISTRY.get(topic)
                    reasoning = '- '.join(topic_data['reasoning_points']) if topic_data else ''
                    specs = topic_data['specs_and_tooling_markdown'] if topic_data else 'Refer to manual'
                    
                    context = f"Topic: {topic}\nReasoning:\n{reasoning}\n\nSpecs:\n{specs}\n\nREFERENCE EXTRACTS:\n{context_str}\n\nQuery: {user_query}"
                    
                    response = call_llm(system_instructions, context)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.rerun()