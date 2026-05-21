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

SPEC_REGISTRY = {
    "OIL CHANGE / MAGNETIC PLUG INSPECTION": """
    MANDATORY MECHANICAL TRUTHS & FLUID SPECIFICATIONS:
    - Removal Phase: Use standard hand wrenches or sockets. NEVER use a torque wrench to loosen fasteners.
    - Fluid Condition: Oil must be drained when warm/hot (operating temperature) to ensure proper scavenging.
    - Approved Oil Type: 4-stroke motorcycle or aviation engine oils meeting Rotax Standard RON 424 (e.g., AeroShell Sport Plus 4).
    - Oil Refill Quantity: Requires approximately 3.0 Litres. Verify final levels using the dipstick after venting.
    - Oil Tank Drain Screw: Requires 17mm socket. Tightening torque is strictly **25 Nm (221 in. lb)**. Fit a new copper ring.
    - Crankcase Magnetic Plug: Requires 24mm socket. Tightening torque is strictly **20 Nm (177 in. lb)**. Never torque to 25 Nm or 30 Nm. Absolute ban on crush washers, gaskets, or thread sealants (no Loctite). Lubricate threads with clean engine oil only.
    - Oil Filter: Part No. 825601. Lube gasket with engine oil, hand tighten 3/4 turn, or torque to **14 Nm (124 in. lb)** using cup tool.
    """,
    "SPARK PLUG INSPECTION": """
    MANDATORY MECHANICAL TRUTHS & SPARK PLUG SPECIFICATIONS:
    - Tool Dimension: Requires a 16mm (5/8") thin-wall spark plug socket.
    - Reinstallation Torque: Tighten strictly to **16 Nm (142 in. lb)** on a cold engine casing.
    - Electrode Clearance: New plug gap is **0.8 mm to 0.9 mm**. Absolute maximum wear limit is **1.1 mm**. Do not manually bend tabs.
    - Sealing Pastes: Minimal film of silicone heat-conduction paste strictly on upper engagement threads. Keep electrodes dry.
    """,
    "OIL PRESSURE CHECK": """
    MANDATORY DIAGNOSTIC PARAMETERS:
    - Testing Method: Connect a calibrated mechanical master pressure gauge into the main oil pump gallery block port via M10x1 adaptor.
    - Hydraulic Limits: Minimum hot idle is **0.8 bar (11.6 psi)**. Normal operation is **2.0 to 5.0 bar (29 to 73 psi)**. Peak cold cap is **7.0 bar (102 psi)**.
    """,
    "CARBURETOR SYNCHRONIZATION": """
    MANDATORY PNEUMATIC SYNCHRONIZATION VALUES & CHECKPOINTS:
    - Safety First: Stay well clear of the spinning propeller arc during running adjustments. Secure the aircraft wheels firmly.
    - Pre-Requisite: Mechanical synchronization (cable slack adjustments) must be performed on a completely cold, non-running engine first.
    - Idle Balancing Limits: Use a calibrated electronic differential pressure gauge (e.g., Carbmate / Synchro) or matching vacuum gauges. Maximum permissible pneumatic pressure variation between carburetor heads at an operating idle of 1800-2000 RPM is strictly **20 mbar (0.29 psi / 0.59 inHg)**. 
    - Cruise Balancing Limits: Crosscheck synchronization at cruise power thresholds (3500 to 4000 RPM). Permissible variation here is strictly **0 mbar** variance (perfect alignment) via precise cable play adjustments.
    - Bowden Cables: Verify all throttle cables possess a minimum free play of **1 mm (0.04 in)** when throttle levers are pinned against the physical idle stops.
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
# 7. OPENROUTER HANDSHAKE (TEMPERATURE 0.2 FOR CADENCE)
# =====================================================
def call_llm(prompt: str):
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "meta-llama/llama-3.1-8b-instruct",
        "temperature": 0.2,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a Senior iRMT LAA/BMAA Inspector and aircraft workshop mentor. Your tone is natural, "
                    "supportive, technically precise, and conversational. You do not talk like a generic bulleted engine list; "
                    "you explain the step mechanics clearly to help the user complete the maintenance activity safely."
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
# 10. USER COMMAND RUNNER WITH TOPIC FLUSH HOOKS
# =====================================================
user_query = st.chat_input("Enter technical maintenance question...")

if user_query:
    current_time = time.time()
    time_passed = current_time - st.session_state.last_query_time
    col_ctx = st.container() if is_admin_mode else center_console
    with col_ctx:
        with st.chat_message("user"): st.write(user_query)

    if st.session_state.active_engine is None:
        engine_match = re.search(r'(912\s*uls|912\s*ul|912\s*is|914|915\s*is|915|916\s*is|916)', user_query.lower())
        if engine_match:
            st.session_state.active_engine = engine_match.group(1).upper().replace(" ", "")
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.session_state.messages.append({"role": "assistant", "content": f"### 🔓 WORKSPACE UNLOCKED\nEngine profile securely set to **ROTAX {st.session_state.active_engine}**."})
            st.rerun()
        else: st.stop()

    # DYNAMIC TOPIC STATE FLUSHING LAYER (Prevents sticky oil context leaking into carb queries)
    if any(w in user_query.lower() for w in ["carb", "sync", "balance", "float", "choke"]):
        st.session_state.active_topic = "CARBURETOR SYNCHRONIZATION"
    elif any(w in user_query.lower() for w in ["plug", "gap", "spark"]):
        st.session_state.active_topic = "SPARK PLUG INSPECTION"
    elif any(w in user_query.lower() for w in ["pressure", "gauge"]) and "oil" in user_query.lower():
        st.session_state.active_topic = "OIL PRESSURE CHECK"
    elif any(w in user_query.lower() for w in ["drain", "magnet", "change", "oil"]):
        st.session_state.active_topic = "OIL CHANGE / MAGNETIC PLUG INSPECTION"

    if time_passed < COOLDOWN_SECONDS or len(user_query) > MAX_QUERY_CHARACTERS or st.session_state.daily_token_consumption >= DAILY_TOKEN_BUDGET:
        st.error("Guardrail condition triggered.")
        st.stop()

    st.session_state.last_query_time = current_time
    st.session_state.messages.append({"role": "user", "content": user_query})

    assistant_canvas = st.chat_message("assistant") if is_admin_mode else col_ctx.chat_message("assistant")
    with assistant_canvas:
        response_placeholder = st.empty()
        if invalid_configuration(user_query, st.session_state.active_engine):
            st.error("Incompatible carburetration configuration engine block.")
            st.stop()
        else:
            with st.spinner("Executing mathematical spatial context scan..."):
                try:
                    context_str, citations_map = "No matching data found.", {}
                    search_query = f"{st.session_state.active_topic or ''} {user_query}"

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

                    # Grab context-specific rules
                    active_truth_injection = SPEC_REGISTRY.get(st.session_state.active_topic, "GENERAL RULES: Provide exact dimensions and tolerances where available.")

                    final_prompt = f"""You are actively mentoring an aircraft technician working on a ROTAX {st.session_state.active_engine}.
                    You must build your response using the manual extracts combined with the mandatory engineering truth matrix rules below.

                    {active_truth_injection}

                    STRICT COGNITIVE SEPARATION RULES:
                    1. TOPIC CONTEXT ISOLATION: The technician's query may have changed. Read the 'TECHNICIAN'S QUERY' line carefully. If the user is asking about CARBURETORS, you are strictly FORBIDDEN from using or talking about oil tank drain plug torques, copper washers, or oil filters. Keep your workspace completely clean.
                    2. CRITICAL SPECIFICATION MATCHING: If the Master truths state a value (e.g., Magnetic plug torque is 20 Nm), you must print EXACTLY that value. Never allow data snippets from the extracts to change a hardcoded truth value.
                    3. FLUID TOOL LOGIC: Plugs and adjusters are threaded fasteners. They are turned with wrench or socket tools. Never say they are removed by hand. Torque wrenches apply strictly to final reassembly tightening.
                    4. TWO-STROKE INFORMATION IS COMPLETELY BANNED.

                    Structure your response exactly like this to maintain an authoritative, mentor voice:

                    ### 1. THE WORKBENCH PROCEDURE
                    * Provide a smooth, logically phased walkthrough of the exact task requested. Explain the mechanical reasoning behind why a step or tolerance matters.

                    ### 2. ⚠️ INSPECTOR'S SAFETY BRIEF
                    * Speak directly to the technician. Warn them about critical failure modes or high-risk blunders specific to this exact procedure.

                    ### 3. REQUIRED PARTS & SPECIFICATIONS
                    * Itemize verified part codes, tool constraints, and clear tolerance dimensions required on their shop tray.
                    ---
                    REFERENCE EXTRACTS: {context_str}
                    TECHNICIAN'S QUERY: {user_query}"""

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