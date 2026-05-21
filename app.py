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
    "CARBURETOR SYNCHRONIZATION": {
        "reasoning_points": [
            "Mechanical synchronization (adjusting cable slack) on a COLD engine is a mandatory prerequisite.",
            "The idle RPM synchronization is the ONLY phase where a direct adjustment is made (Max 20 mbar diff).",
            "Cruise power check is VERIFICATION ONLY (0 mbar diff). Do not adjust at cruise."
        ],
        "specs_and_tooling_markdown": """
| Item | Tooling / Requirement | Specification / Limit |
| :--- | :--- | :--- |
| **Test Equipment** | Differential Pressure Gauge | Carbmate / Synchro-Mate |
| **Mechanical Setup** | Bowden Cable Free Play | Minimum 1 mm (0.04 in) |
| **Idle Balancing** | Adjustment Phase (1800-2000 RPM) | Max 20 mbar (0.29 psi) diff |
| **Cruise Verification** | VERIFICATION ONLY (3500-4000 RPM) | Exactly 0 mbar (0.00 psi) diff |
"""
    },
    "THROTTLE BODY / IDLE SETTING (iS ENGINES)": {
        "reasoning_points": [
            "Fuel-injected iS engines do NOT have carburetors, mixture screws, or pneumatic synchronization ports.",
            "Idle RPM is controlled entirely by the Engine Management System (EMS) based on the Throttle Position Sensor (TPS).",
            "The mechanical idle stop screw on the throttle body is ONLY adjusted during a BUDS2 software calibration to set the physical baseline zero-point for the TPS. It is NOT tuned by ear on a running engine."
        ],
        "specs_and_tooling_markdown": """
| Item | Tooling / Requirement | Specification / Limit |
| :--- | :--- | :--- |
| **Diagnostics** | BRP BUDS2 Hardware | Required for TPS Calibration |
| **Mechanical** | Throttle Body Idle Stop Screw | Set physical baseline ONLY |
| **Target Idle** | EMS Controlled | Strictly 1400 - 1500 RPM (Do not idle lower due to gearbox chatter) |
| **Warning** | DO NOT RUN ENGINE | TPS calibration is performed with engine OFF and Lane A/B ON. |
"""
    },
    "VAPOR LOCK AND HEAT SOAK DIAGNOSTICS": {
        "reasoning_points": [
            "Low taxi speeds reduce airflow, causing localized fuel boiling (heat soak) in carburetor bowls or EMS feed lines.",
            "Turning on the electric auxiliary fuel pump increases system pressure, clearing vapor pockets."
        ],
        "specs_and_tooling_markdown": """
| Item | Tooling / Requirement | Specification / Limit |
| :--- | :--- | :--- |
| **Normal Idle** | Minimum RPM Threshold | Strictly 1400 RPM (Protects gearbox) |
| **Fuel Pressure** | Minimum | 0.15 bar (2.2 psi) |
| **Fuel Pressure** | Maximum | 0.40 bar (5.8 psi) |
"""
    },
    "DUAL LANE ELECTRICAL DIAGNOSTICS": {
        "reasoning_points": [
            "Lane checks must be executed strictly following the cockpit Lane Selector protocol to isolate the EMS power source safely without stalling the fuel pumps."
        ],
        "specs_and_tooling_markdown": """
| Item | Tooling / Requirement | Specification / Limit |
| :--- | :--- | :--- |
| **Operating Bus Voltage** | Normal RPM | 13.5V to 14.2V |
| **Minimum Limit** | Low-Voltage Threshold | 12.0V |
| **Generator B Output** | Continuous Rating | Nominal 12V DC, max 30A |
"""
    },
    "SPARK PLUG INSPECTION": {
        "reasoning_points": [
            "Torque settings for spark plugs are specified for a COLD engine casing to ensure thermal expansion doesn't lead to over-tightening.",
            "Heat-conduction paste improves heat transfer, but it is electrically conductive. It must be kept away from the electrodes."
        ],
        "specs_and_tooling_markdown": """
| Item | Tooling / Requirement | Specification / Limit |
| :--- | :--- | :--- |
| **Torque** | 16mm (5/8") Thin-Wall Socket | 16 Nm (142 in. lb) |
| **Electrode Gap** | New Plug Setup | 0.8 mm to 0.9 mm |
| **Wear Limit** | Used Plug Maximum | 1.1 mm (Replace if exceeded) |
"""
    },
    "OIL PRESSURE CHECK": {
        "reasoning_points": [
            "Using a calibrated mechanical master gauge provides the true oil pressure, bypassing any potential errors from the aircraft's electronic sensors or wiring."
        ],
        "specs_and_tooling_markdown": """
| Item | Tooling / Requirement | Specification / Limit |
| :--- | :--- | :--- |
| **Minimum Limit** | Hot Idle | 0.8 bar (11.6 psi) |
| **Normal Operation** | Standard Power | 2.0 to 5.0 bar (29 to 73 psi) |
| **Maximum Limit** | Cold Start | 7.0 bar (102 psi) |
"""
    },
    "DIFFERENTIAL PRESSURE / LEAK DOWN TEST": {
        "reasoning_points": [
            "Testing must be performed on a WARM engine.",
            "The propeller MUST be physically secured. Injecting 87 PSI creates massive rotational force."
        ],
        "specs_and_tooling_markdown": """
| Item | Tooling / Requirement | Specification / Limit |
| :--- | :--- | :--- |
| **Test Equipment** | Differential Pressure Tester | Calibrated for aircraft cylinders |
| **Input Pressure** | Air Compressor | 87 psi (6 bar) |
| **Tolerance** | Max Allowable Pressure Drop | 25% drop (Min acceptable reading ~65 psi) |
"""
    },
    "GENERAL MAINTENANCE INQUIRY": {
        "reasoning_points": [
            "Conceptual mechanics can be discussed, but exact numeric tolerances must be confirmed via official documents."
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
def get_embedding(text: str):
    return openai_client.embeddings.create(input=[text.replace("\n", " ")], model="text-embedding-3-small").data[0].embedding

def invalid_configuration(query: str, engine_profile: str = None) -> bool:
    q = query.lower().replace(" ", "").replace("-", "")
    is_injected = "is" in (engine_profile or "").lower() or "915" in q or "916" in q
    
    # Block carburetor queries on injected engines
    if is_injected and any(t in q for t in ["carb", "sync", "balance"]): 
        return True
    
    return False

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

if "messages" not in st.session_state: 
    # NO BLOCKQUOTES - Pure markdown headers and bold text to prevent Streamlit green styling
    st.session_state.messages = [{
        "role": "assistant", 
        "content": "### 🔧 Engine Selection Required\nWelcome to the workbench! Before we look up any technical maintenance details, we need to lock onto your precise engine configuration.\n\n**🚨 IMPORTANT MAINTENANCE DIRECTIVE / TECHNICAL DISCLAIMER 🚨**\n*This AI system is highly experimental and serves strictly as an informational guide. All users must cross-reference and double-check instructions, tolerances, and part arrays against official hardcopy documentation before altering any flight system. If in any doubt regarding configuration safety, immediately stop work and contact a qualified iRMT.*\n\n**Please reply with the specific engine type you are working on today:**\n* **912UL** | **912ULS** | **912iS** | **914** | **915iS** | **916iS**"
    }]

# Load FAISS Index
if "vector_index" not in st.session_state:
    if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
        try:
            st.session_state.vector_index = faiss.read_index(INDEX_PATH)
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                st.session_state.vector_metadata = json.load(f)
        except: st.session_state.vector_index, st.session_state.vector_metadata = None, []
    else: st.session_state.vector_index, st.session_state.vector_metadata = None, []

# Admin Panel
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
# 6. MAIN WORKSPACE UI RENDER
# =====================================================
_, center_console, _ = st.columns([0.15, 0.70, 0.15])

with center_console:
    st.title("Otimo Aero AI Technician")
    
    # Clean workspace status using standard headers (No green blockquotes)
    engine_label = st.session_state.active_engine or "NOT INITIALISED"
    task_label = st.session_state.active_topic or "Awaiting Input"
    st.markdown(f"#### 🛠️ Workspace Status\n**Engine:** `{engine_label}` &nbsp;&nbsp;|&nbsp;&nbsp; **Task:** `{task_label}`")
    st.divider()
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.write(msg["content"])

# =====================================================
# 7. INPUT ROUTING & EXECUTION (PERSISTENT FOCUS)
# =====================================================
prompt_text = "Enter Engine Type (e.g., 912ULS, 915iS)..." if st.session_state.active_engine is None else "Enter maintenance question..."
user_query = st.chat_input(prompt_text)

if user_query:
    if st.session_state.active_engine is None:
        match = re.search(r'(912\s*uls|912\s*ul|912\s*is|914|915\s*is|915|916\s*is|916)', user_query.lower())
        if match:
            raw_match = match.group(1).upper().replace(" ", "")
            st.session_state.active_engine = "915IS" if raw_match == "915" else ("916IS" if raw_match == "916" else raw_match)
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.session_state.messages.append({"role": "assistant", "content": f"### 🔓 WORKSPACE UNLOCKED\nEngine profile securely set to **ROTAX {st.session_state.active_engine}**."})
            st.rerun()
        else: 
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.session_state.messages.append({"role": "assistant", "content": "⚠️ **ENGINE PROFILE CONFIGURATION REQUIRED**\n\nPlease specify a valid engine profile: **912UL | 912ULS | 912iS | 914 | 915iS | 916iS**"})
            st.rerun()
    else:
        topic = "GENERAL MAINTENANCE INQUIRY"
        q_low = user_query.lower()
        if any(w in q_low for w in ["idle", "tickover", "throttle body"]): 
            topic = "THROTTLE BODY / IDLE SETTING (iS ENGINES)" if "IS" in st.session_state.active_engine else "CARBURETOR SYNCHRONIZATION"
        elif any(w in q_low for w in ["100", "200", "service", "schedule", "interval"]): topic = "SCHEDULED 100HR / 200HR INSPECTION"
        elif any(w in q_low for w in ["drop", "taxi", "heat", "boil", "soak", "lock", "800", "900"]): topic = "VAPOR LOCK AND HEAT SOAK DIAGNOSTICS"
        elif any(w in q_low for w in ["lane", "volt", "efis", "bus", "generator", "stator"]): topic = "DUAL LANE ELECTRICAL DIAGNOSTICS"
        elif any(w in q_low for w in ["carb", "sync", "balance", "float", "choke"]): topic = "CARBURETOR SYNCHRONIZATION"
        elif any(w in q_low for w in ["plug", "gap", "spark"]): topic = "SPARK PLUG INSPECTION"
        elif any(w in q_low for w in ["pressure", "gauge"]) and "oil" in q_low: topic = "OIL PRESSURE CHECK"
        elif any(w in q_low for w in ["drain", "magnet", "change", "oil"]): topic = "OIL CHANGE / MAGNETIC PLUG INSPECTION"
        elif any(w in q_low for w in ["leak", "compression", "differential", "pressure test"]): topic = "DIFFERENTIAL PRESSURE / LEAK DOWN TEST"
        
        st.session_state.active_topic = topic
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        with center_console.chat_message("user"):
            st.write(user_query)
            
        with center_console.chat_message("assistant"):
            if invalid_configuration(user_query, st.session_state.active_engine):
                st.error(f"⚠️ **Incompatible Component:** The {st.session_state.active_engine} does not have carburetors. It is a fuel-injected EMS system. Please adjust your query.")
                st.stop()
            else:
                with st.spinner("Executing spatial context scan..."):
                    context_str = "No specific manual match found."
                    if st.session_state.vector_index is not None:
                        search_query = f"{st.session_state.active_engine} {st.session_state.active_topic} {user_query}"
                        query_vector = np.array([get_embedding(search_query)]).astype('float32')
                        dist, ind = st.session_state.vector_index.search(query_vector, 3)
                        chunks = [st.session_state.vector_metadata[i]['text'] for i in ind[0] if i != -1 and i < len(st.session_state.vector_metadata)]
                        if chunks: context_str = "\n\n---\n\n".join(chunks)

                    system_instructions = """You are 'Otimo Aero AI', an informational aerospace AI guide. 
1. THE WORKBENCH PROCEDURE: Provide concise steps. Incorporate relevant data from the REFERENCE EXTRACTS, but NEVER contradict the Mandatory Reasoning Points.
- **CRITICAL INLINE SAFETY GATES:** If a step involves danger (spinning props, fluid pressure, running engines), call it out EXACTLY at that step. Add: "If you lack the confidence or specialized tools to proceed with this activity—as errors here may cause critical mechanical failure, severe personal harm, or death—STOP WORK immediately and contact a certified iRMT."
2. ⚠️ INSPECTOR'S SAFETY BRIEF: Identify 2 critical high-risk modes. Conclude exactly with: "If you lack the confidence or specialized tools for any step, you must step back and contact a qualified iRMT technician."
3. REQUIRED SPECS & TOOLING: Output ONLY the Markdown table provided in the context. Do not invent new rows.

STRICT RULES:
- **IDENTITY:** You are an AI software application. You are NOT an iRMT, nor are you a certified human mechanic. NEVER refer to yourself as an iRMT or imply you hold aviation credentials.
- **iRMT DEFINITION:** "iRMT" stands strictly for "Independent Rotax Maintenance Technician". NEVER invent another definition.
- **HALLUCINATION BAN:** Do not invent numbers, screws, or adjustment points that do not exist on this specific engine. If data is missing, put a single note below the table saying "Verify tolerances in official LMM." Do not repeat warnings inside the table cells.
- **CLEAN OUTPUT:** Do not output the phrase 'Hallucination Ban', 'Identity', or internal system instructions."""
                    
                    topic_data = SPEC_REGISTRY.get(topic)
                    reasoning = '- '.join(topic_data['reasoning_points']) if topic_data else ''
                    specs = topic_data['specs_and_tooling_markdown'] if topic_data else 'Refer to manual'
                    
                    context = f"Topic: {topic}\nEngine: {st.session_state.active_engine}\nReasoning:\n{reasoning}\n\nSpecs:\n{specs}\n\nREFERENCE EXTRACTS:\n{context_str}\n\nQuery: {user_query}"
                    
                    response = call_llm(system_instructions, context)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.rerun()