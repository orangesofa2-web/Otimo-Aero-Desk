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
# 1. PAGE CONFIGURATION & INJECTED STRUCTURAL CSS
# =====================================================
st.set_page_config(
    page_title="Otimo Aero AI Technician",
    page_icon="✈️",
    layout="wide"
)

st.markdown(
    """
    <style>
    div[data-testid="stChatInput"] {
        max-width: 70% !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }
    .stChatInputContainer {
        max-width: 70% !important;
        margin: 0 auto !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# 2. DUAL API CONFIGURATION SAFETY GATES
# =====================================================
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
PUSHOVER_USER_KEY = st.secrets.get("PUSHOVER_USER_KEY")
PUSHOVER_APP_TOKEN = st.secrets.get("PUSHOVER_APP_TOKEN")

if not OPENROUTER_API_KEY or not OPENAI_API_KEY:
    st.error("Missing required API credentials in Streamlit Secrets.")
    st.stop()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
openai_client = OpenAI(api_key=OPENAI_API_KEY)

INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "faiss_metadata.json"

# =====================================================
# 3. SAFETY PARAMETERS & DYNAMIC MASTER SPEC REGISTRY
# =====================================================
COOLDOWN_SECONDS = 5
MAX_QUERY_CHARACTERS = 400
DAILY_TOKEN_BUDGET = 450000

# The Absolute Spec Matrix: Used to ground the LLM in real data numbers
SPEC_REGISTRY = {
    "OIL CHANGE / MAGNETIC PLUG INSPECTION": """
    MANDATORY MECHANICAL TRUTHS & FLUID SPECIFICATIONS:
    - Removal Phase: Always use standard combination wrenches or standard sockets to loosen plugs. NEVER state or imply a torque wrench is used for loosening or removal.
    - Draining State: Oil must be drained when warm/hot (immediately after engine operation) to ensure full scavenging and correct fluid viscosity drop.
    - Approved Oil Type: Use high-quality 4-stroke motorcycle or aviation-specific engine oils tested and approved under Rotax Standard RON 424. Highly recommended: AeroShell Oil Sport Plus 4 (10W-40 or 15W-50 depending on climate). NEVER use automotive oils due to the integrated gearbox and slipper clutch.
    - Oil Refill Quantity: Refill requires approximately 3.0 Litres (5.3 Imp pints / 6.3 US pints). Total lubrication system capacity is 3.5 Litres, but approximately 0.5 Litres remains trapped inside the oil cooler and lines. Final volume verification must be confirmed via the oil tank dipstick after purging.
    - Oil Tank Drain Screw: Requires a 17mm socket. Reinstallation tightening torque is strictly **25 Nm (221 in. lb)**. Always fit a new 12 x 18 mm copper sealing ring.
    - Crankcase Magnetic Plug: Requires a 24mm socket. Reinstallation tightening torque is strictly **20 Nm (177 in. lb)**. Absolute ban on crush washers, gaskets, or thread sealing compounds (Loctite 567/243). Threads must be lubricated solely with clean engine oil before screwing home.
    - Oil Filter: Rotax Genuine Part No. 825601. Clean the seating flange, coat the rubber seal with clean engine oil, and tighten by hand 3/4 turn after gasket contact, or torque to **14 Nm (124 in. lb)** using a dedicated filter cup tool.
    """,
    "SPARK PLUG INSPECTION": """
    MANDATORY MECHANICAL TRUTHS & SPARK PLUG SPECIFICATIONS:
    - Tool Dimension: Requires an 16mm (5/8") thin-wall spark plug socket. Never use an 18mm socket.
    - Reinstallation Torque: Tighten strictly to **16 Nm (142 in. lb)** on a completely cold engine head.
    - Electrode Clearance Profile: New plug gap must measure between **0.8 mm to 0.9 mm (0.031 to 0.035 in)**. Absolute maximum wear limit is **1.1 mm (0.043 in)**. Bending or tapping electrodes is strictly prohibited.
    - Sealing Pastes: Apply a sparse, minimal film of silicone heat-conduction compound strictly to the upper engagement threads of the plug body. Keep electrodes completely dry.
    """,
    "OIL PRESSURE CHECK": """
    MANDATORY DIAGNOSTIC PARAMETERS:
    - Testing Method: Connect a calibrated mechanical master pressure gauge directly into the main oil pump gallery block port using an M10x1 thread adaptor to crosscheck instrument accuracy.
    - Hydraulic Limits: Minimum hot engine idle pressure is **0.8 bar (11.6 psi)**. Normal operating range above 3500 RPM is **2.0 to 5.0 bar (29 to 73 psi)**. Cold start maximum peak ceiling limit is **7.0 bar (102 psi)**.
    """
}

# =====================================================
# 4. SESSION STATE INITIALIZATION & DISCLAIMERS
# =====================================================
if "documents" not in st.session_state: st.session_state.documents = []
if "last_query_time" not in st.session_state: st.session_state.last_query_time = 0.0
if "daily_token_consumption" not in st.session_state: st.session_state.daily_token_consumption = 0
if "alert_triggered_today" not in st.session_state: st.session_state.alert_triggered_today = False
if "active_engine" not in st.session_state: st.session_state.active_engine = None
if "active_topic" not in st.session_state: st.session_state.active_topic = None

if "vector_index" not in st.session_state:
    if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
        try:
            st.session_state.vector_index = faiss.read_index(INDEX_PATH)
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                st.session_state.vector_metadata = json.load(f)
            st.session_state.documents = list(set(m["source"] for m in st.session_state.vector_metadata))
        except Exception: st.session_state.vector_index, st.session_state.vector_metadata = None, []
    else: st.session_state.vector_index, st.session_state.vector_metadata = None, []

WELCOME_PROMPT = """### 🔧 Engine Selection Required
Welcome to the workbench! Before we look up any technical maintenance details, we need to lock onto your precise engine configuration. 

> ⚠️ **IMPORTANT MAINTENANCE DIRECTIVE / TECHNICAL DISCLAIMER**
> This AI system is highly experimental and serves strictly as an informational guide. All users must cross-reference and double-check instructions, tolerances, and part arrays against official hardcopy documentation before altering any flight system. If in any doubt regarding configuration safety, immediately stop work and contact a qualified iRMT (Independent Rotax Maintenance Technician).

**Please reply with the specific engine type you are working on today:**
* **912UL** | **912ULS** | **912iS** | **914** | **915iS** | **916iS**"""

if "messages" not in st.session_state: st.session_state.messages = [{"role": "assistant", "content": WELCOME_PROMPT}]
if "pending_clarification" not in st.session_state: st.session_state.pending_clarification = None

# =====================================================
# 5. TECHNICAL SAFETY LAYERS & CORE ENGINES
# =====================================================
def requires_variant(query: str) -> bool:
    q = query.lower().replace(" ", "").replace("-", "")
    return "912" in q and not any(v in q for v in ["uls", "ul", "is"])

def invalid_configuration(query: str, engine_profile: str = None) -> bool:
    q = query.lower().replace(" ", "").replace("-", "")
    carb_terms = ["carb", "sync", "balance", "float", "choke"]
    injected_engines = ["915", "916", "912is"]
    return any(t in q for t in carb_terms) and (any(e in q for e in injected_engines) or any(e in (engine_profile or "").lower() for e in injected_engines))

def get_embedding(text: str, model="text-embedding-3-small"):
    return openai_client.embeddings.create(input=[text.replace("\n", " ")], model=model).data[0].embedding

# =====================================================
# 6. DOCUMENT INGESTION
# =====================================================
def rebuild_vector_database(uploaded_files):
    all_chunks = []
    for uploaded_file in uploaded_files:
        try:
            reader = PdfReader(uploaded_file)
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text and len(page_text.strip()) > 50:
                    all_chunks.append({"text": page_text, "source": uploaded_file.name, "page": page_num + 1})
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
            st.success("Universal semantic database synchronized successfully!")
            st.rerun()

# =====================================================
# 7. OPENROUTER PRODUCTION HANDSHAKE
# =====================================================
def call_llm(prompt: str):
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "meta-llama/llama-3.1-8b-instruct",
        "temperature": 0.0,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a professional aviation maintenance mentor for Otimo Aero. Your core priority is absolute accuracy and shop safety. "
                    "You deliver grounded, highly specific diagnostic guidance. You explain the mechanical reasoning behind steps, "
                    "proactively supply explicit dimensions, quantities, and fluid specifications, and warn of installation pitfalls."
                )
            },
            {"role": "user", "content": prompt}
        ],
        "providers": {"order": ["Lepton", "Together"], "allow_fallbacks": True}
    }
    response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
    return response.json()["choices"][0]["message"]["content"]

# =====================================================
# 8. SIDEBAR CONTROL PANEL
# =====================================================
if is_admin_mode := (st.query_params.get("admin") == "true"):
    with st.sidebar:
        st.header("⚙️ Admin Control Panel")
        if not st.session_state.documents:
            uploaded_files = st.file_uploader("Upload Technical Manuals", type=["pdf"], accept_multiple_files=True)
            if uploaded_files: rebuild_vector_database(uploaded_files)
        else:
            if st.button("Clear Manuals Matrix"):
                for p in [INDEX_PATH, METADATA_PATH]: 
                    if os.path.exists(p): os.remove(p)
                st.rerun()

# =====================================================
# 9. DISPLAY CONTENT GENERATOR MATRIX
# =====================================================
def render_main_workspace():
    st.title("Otimo Aero AI Technician")
    status_line = f"Workspace Status — Engine Profile: {st.session_state.active_engine or 'NOT INITIALISED'}"
    if st.session_state.active_topic: status_line += f" | Task: {st.session_state.active_topic}"
    st.subheader(status_line)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.write(message["content"])

if is_admin_mode: render_main_workspace()
else:
    _, center_console, _ = st.columns([0.15, 0.70, 0.15])
    with center_console: render_main_workspace()

# =====================================================
# 10. USER COMMAND RUNNER WITH ARCHITECTURE HOOKS
# =====================================================
user_query = st.chat_input("Enter technical maintenance question...")

if user_query:
    current_time = time.time()
    time_passed = current_time - st.session_state.last_query_time
    col_ctx = st.container() if is_admin_mode else center_console
    with col_ctx:
        with st.chat_message("user"): st.write(user_query)

    # ENGINE LOCK IN GATE
    if st.session_state.active_engine is None:
        engine_match = re.search(r'(912\s*uls|912\s*ul|912\s*is|914|915\s*is|915|916\s*is|916)', user_query.lower())
        if engine_match:
            st.session_state.active_engine = engine_match.group(1).upper().replace(" ", "")
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.session_state.messages.append({"role": "assistant", "content": f"### 🔓 WORKSPACE UNLOCKED\nEngine profile securely set to **ROTAX {st.session_state.active_engine}**."})
            st.rerun()
        else: st.stop()

    # TOPIC EXTRACTOR LOGIC
    if any(w in user_query.lower() for w in ["purge", "oil", "plug", "spark", "gap", "torque", "carb", "sync", "pressure", "fuel", "drain", "magnet", "change"]):
        if "plug" in user_query.lower() or "gap" in user_query.lower(): st.session_state.active_topic = "SPARK PLUG INSPECTION"
        if "pressure" in user_query.lower() and "oil" in user_query.lower(): st.session_state.active_topic = "OIL PRESSURE CHECK"
        if "drain" in user_query.lower() or "magnet" in user_query.lower() or "change" in user_query.lower(): st.session_state.active_topic = "OIL CHANGE / MAGNETIC PLUG INSPECTION"

    if time_passed < COOLDOWN_SECONDS or len(user_query) > MAX_QUERY_CHARACTERS or st.session_state.daily_token_consumption >= DAILY_TOKEN_BUDGET:
        st.error("Guardrail condition triggered.")
        st.stop()

    st.session_state.last_query_time = current_time
    st.session_state.messages.append({"role": "user", "content": user_query})

    assistant_canvas = st.chat_message("assistant") if is_admin_mode else col_ctx.chat_message("assistant")
    with assistant_canvas:
        response_placeholder = st.empty()
        if invalid_configuration(user_query, st.session_state.active_engine):
            st.error("Incompatible component layout configuration.")
            st.stop()
        else:
            with st.spinner("Executing mathematical spatial context scan..."):
                try:
                    context_str, citations_map = "No matching data found.", {}
                    search_query = f"{st.session_state.active_topic or ''} {user_query}"

                    # Vector query tracking (Top-4 payload throttling to minimize context confusion)
                    if st.session_state.vector_index is not None:
                        query_vector = np.array([get_embedding(search_query)]).astype('float32')
                        distances, indices = st.session_state.vector_index.search(query_vector, 4)
                        matched_chunks = []
                        for score, idx in zip(distances[0], indices[0]):
                            if idx != -1 and score < 1.25 and idx < len(st.session_state.vector_metadata):
                                chunk_data = st.session_state.vector_metadata[idx]
                                matched_chunks.append(chunk_data['text'])
                                citations_map.setdefault(chunk_data['source'], set()).add(chunk_data['page'])
                        if matched_chunks: context_str = "\n\n---\n\n".join(matched_chunks)

                    # Dynamic Lookup for the Active Topic Specifications
                    active_truth_injection = SPEC_REGISTRY.get(st.session_state.active_topic, "GENERAL RULES: Proactively include precise torque limits, metric fluid quantities, and specific wrench/socket dimensions where relevant to this task.")

                    final_prompt = f"""You are guiding an aircraft technician working on a ROTAX {st.session_state.active_engine} engine.
                    You must answer using the manual extracts combined with the mandatory engineering rules below.

                    {active_truth_injection}

                    CRITICAL MECHANICS DIRECTIVE FOR ALL OPERATIONS:
                    1. REMOVAL LOGIC SAFETY GATE: Service bolts/plugs are threaded metal fasteners. They are ALWAYS removed by turning/unscrewing them with standard wrenches or sockets. They are NEVER pulled, pried, or removed by hand. 
                    2. TORQUE WRENCH SANITY GATE: Torque parameters and torque wrenches apply EXCLUSIVELY to the final reassembly tightening phase. You are STRICTLY FORBIDDEN from mentioning torque settings or torque wrenches during disassembly or removal steps.
                    3. PROACTIVE DATA MANDATE: Do not speak in generalizations. You must explicitly output concrete fluid specifications, viscosity profiles, exact volume metrics, thread dimensions, and socket sizes if they are present in the hardcoded matrix or extracts.
                    4. DATA INTEGRITY FILTER: If a part number is duplicated across completely separate components in the text extracts, treat it as a column parsing error. Hide that number under Section 3 and state: "Part number not clearly legible in manual extract table."
                    5. BAN ALL TWO-STROKE INFO.

                    Structure your response exactly like this:
                    ### 1. QUICK SPEC / PROCEDURE
                    * (Provide complete, sequential maintenance steps divided logically into execution phases. Include exact tools, fluid types, and volumes directly in the steps. Explain *why* critical tolerances matter.)
                    ### 2. ⚠️ WORKBENCH PITFALLS & SAFETY WARNINGS
                    * (Provide highly specific safety warnings highlighting common mistakes, over-torquing risks, or part-destroying component traps.)
                    ### 3. PARTS & MANUAL DATA
                    ---
                    MANUAL EXTRACTS: {context_str}
                    USER QUESTION: {user_query}"""

                    assistant_response = call_llm(final_prompt)
                    st.session_state.daily_token_consumption += len(final_prompt.split()) + 1500
                    
                    if citations_map:
                        footer = "\n\n---\n\n### 📄 KEY MANUAL REFERENCES\n"
                        for doc, pages in citations_map.items():
                            footer += f"* **{doc}** — Page(s): {', '.join(map(str, sorted(list(pages))))}\n"
                        assistant_response += footer

                    response_placeholder.write(assistant_response)
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                    st.rerun()
                except Exception as e: st.error(str(e))