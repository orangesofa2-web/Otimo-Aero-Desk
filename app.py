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
    "OIL CHANGE / MAGNETIC PLUG INSPECTION": {
        "reasoning_points": [
            "Draining oil when the engine is WARM/HOT is mandatory to ensure wear particles are suspended and drain out completely (scavenging). Cold oil is thick and will not drain fully.",
            "Torque wrenches are precision calibration instruments for TIGHTENING ONLY. Using them for loosening permanently damages their internal mechanism and makes future torque readings unreliable, risking fastener failure.",
            "The Crankcase Magnetic Plug uses a precision TAPERED seat for a metal-to-metal seal. Adding a washer or sealant will prevent a proper seal, causing leaks and potentially damaging the crankcase threads.",
            "The Oil Tank Drain Screw (Sump) uses a soft copper sealing ring that is crushed on tightening. This crushing action creates the seal. It is a one-time-use item and MUST be replaced to prevent leaks.",
            "Lubricating threads and gaskets with clean engine oil prevents thread galling (cold welding) and ensures the applied torque translates into the correct clamping force, not dissipated as friction."
        ],
        "specs_and_tooling_markdown": """
- **Engine Pre-Condition:** Drain the oil only when **WARM or HOT**.
- **Approved Oil Type:** 4-stroke engine oil meeting Rotax Standard RON 424 (e.g., AeroShell Sport Plus 4).
- **Refill Quantity:** Approx. 3.0 Litres (final level MUST be verified via dipstick after a ground run).

---
##### **Component-Specific Data:**

- **Oil Tank Drain Screw (Sump Plug):**
    - **Removal Tooling:** Standard 17mm socket/wrench.
    - **Installation Tooling:** Calibrated torque wrench.
    - **Sealing:** **MUST** use a **NEW** copper sealing ring (one-time use part).
    - **Installation Torque:** **25 Nm (221 in. lb)**.

- **Crankcase Magnetic Plug (Tapered Seal):**
    - **Removal Tooling:** Standard 24mm socket/wrench.
    - **Installation Tooling:** Calibrated torque wrench.
    - **Thread Prep:** Lubricate threads with a light film of clean engine oil before installation.
    - **Installation Torque:** Strictly **20 Nm (177 in. lb)**.
    - **CRITICAL WARNING:** This plug uses a metal-to-metal tapered seal.
        - **DO NOT** use a washer, gasket, or sealing ring of any kind.
        - **DO NOT** apply Loctite, thread sealant, or any other chemical compound.

- **Oil Filter (Part No. 825701 or 825601):**
    - **Tooling:** Standard oil filter wrench for removal and final tightening.
    - **Gasket Prep:** Lubricate rubber gasket with a film of clean engine oil.
    - **Installation:** Hand-tighten until gasket makes contact, then tighten a further **3/4 turn** using the filter wrench.
    - **CRITICAL WARNING:**
        - **DO NOT** use a torque wrench on the oil filter. The specified 3/4 turn method achieves the correct gasket compression.
        - Over-tightening can damage the filter housing or make future removal extremely difficult.

---
##### **UNIVERSAL SAFETY DIRECTIVES:**
- **TORQUE WRENCHES:** Are for **TIGHTENING ONLY**. Never use a torque wrench to loosen fasteners. Use standard hand tools for all removal steps.
- **REUSED PARTS:** The copper sealing ring for the Oil Tank Drain Screw is **NEVER** to be reused.
"""
    },
    "SPARK PLUG INSPECTION": {
        "reasoning_points": [
            "Torque settings for spark plugs are specified for a COLD engine casing to ensure thermal expansion doesn't lead to over-tightening, which can damage the cylinder head threads.",
            "Heat-conduction paste improves heat transfer to the cylinder head, but it is electrically conductive. It must be kept away from the electrodes and center insulator to prevent misfires.",
            "An incorrect electrode gap alters ignition timing and combustion efficiency, leading to poor performance and potential engine damage."
        ],
        "specs_and_tooling_markdown": """
- **Engine Pre-Condition:** Must be a COLD engine for installation/torquing.
- **Required Socket:** 16mm (5/8") thin-wall spark plug socket.
- **Reinstallation Torque (Cold Engine):** Strictly **16 Nm (142 in. lb)**.
- **Electrode Gap (New Plug):** 0.8 mm to 0.9 mm.
- **Maximum Wear Limit (Used Plug):** 1.1 mm (replace plug if exceeded).
- **Sealing Paste:** Minimal film of silicone heat-conduction paste on upper engagement threads ONLY. Keep electrodes and insulator clean and dry.
"""
    },
    "OIL PRESSURE CHECK": {
        "reasoning_points": [
            "Using a calibrated mechanical master gauge provides the true oil pressure, bypassing any potential errors from the aircraft's electronic sensors or wiring.",
            "Pressure checks must be performed at specified RPMs and temperatures to compare against baseline engineering values for a valid diagnosis."
        ],
        "specs_and_tooling_markdown": """
- **Test Equipment:** Calibrated mechanical master pressure gauge with M10x1 adaptor.
- **Connection Point:** Main oil pump gallery block port.
- **Hydraulic Limits:**
    - **Minimum (Hot Idle):** 0.8 bar (11.6 psi)
    - **Normal Operation:** 2.0 to 5.0 bar (29 to 73 psi)
    - **Maximum (Cold Start):** 7.0 bar (102 psi)
"""
    },
    "CARBURETOR SYNCHRONIZATION": {
        "reasoning_points": [
            "Mechanical synchronization (adjusting cable slack) on a COLD engine is a mandatory prerequisite. Pneumatic balancing cannot fix an incorrect mechanical setup.",
            "The idle RPM synchronization is the ONLY phase where a direct adjustment is made based on pneumatic readings. This has a specific tolerance (20 mbar).",
            "The cruise power (3500-4000 RPM) check is a VERIFICATION-ONLY step with a non-negotiable, zero-tolerance (0 mbar) requirement. It is NOT an adjustment point.",
            "Any pressure deviation at cruise power signifies a serious mechanical fault (e.g., cable stretch, bent linkage) that causes destructive harmonic vibrations. The procedure must be stopped and the fault corrected."
        ],
        "specs_and_tooling_markdown": """
- **Safety Pre-Checks:** Secure aircraft, chock wheels, and ensure propeller arc is clear of all personnel and equipment.
- **Test Equipment:** Calibrated electronic differential pressure gauge (e.g., Carbmate, Synchro-Mate).

---
##### **Step 1: Mechanical Synchronization (COLD ENGINE)**
- This is a mandatory prerequisite.
- Verify all throttle Bowden cables possess a **minimum free play of 1 mm (0.04 in)** against the physical idle stops to ensure butterflies are fully closed.

---
##### **Step 2: Pneumatic Balancing (WARM ENGINE)**

###### **Part A: Idle Speed Adjustment (1800-2000 RPM)**
- This is the primary adjustment step.
- The maximum allowable pressure difference between carburetors is **20 mbar (0.29 psi)**.
- Use the idle speed synchronization screw to balance the pressures within this tolerance.

###### **Part B: Cruise Power Verification (3500-4000 RPM)**
- **This is a VERIFICATION step ONLY. NO adjustments are made at this power setting.**
- The required pressure difference between carburetors is **perfectly 0 mbar (0.00 psi)**. There is no acceptable tolerance.

- **CRITICAL SAFETY WARNING:**
    - If the pressure reading is **anything other than 0 mbar** at this cruise power setting, the system is **NOT balanced** and the aircraft is **NOT airworthy**.
    - **DO NOT** attempt to adjust carburetors at this RPM. A non-zero reading indicates a failure in the mechanical linkage, cables, or carburetor components.
    - You MUST return to idle, shut down the engine, and diagnose/correct the underlying mechanical fault before repeating the entire synchronization procedure.
"""
    },
    "VAPOR LOCK AND HEAT SOAK DIAGNOSTICS": {
        "reasoning_points": [
            "Low taxi speeds drastically reduce cowled engine airflow. Exhaust heat radiates upward directly into the Bing carburetor bowls and fuel feed lines, causing localized fuel boiling (heat soak).",
            "When fuel boils inside the delivery circuit, gas bubbles displace liquid fuel. This forces an extreme lean condition at idle, causing a severe RPM drop to 800-900 RPM and violent engine chattering.",
            "Turning on the electric auxiliary fuel pump increases system pressure, raising the boiling point of the fuel and clearing vapor pockets by pushing them back through the return line."
        ],
        "specs_and_tooling_markdown": """
- **Target Engine Profile:** Rotax 912 UL / 912 ULS carburetor configurations.
- **Normal Idle Threshold:** Strictly **1400 RPM** (never allow a 912 series to idle long-term below 1300 RPM due to gearbox dog-clutch chattering).
- **Fuel System Pressure Limits:**
    - **Minimum Fuel Pressure:** **0.15 bar (2.2 psi)**.
    - **Maximum Fuel Pressure:** **0.40 bar (5.8 psi)**.

---
##### **IMMEDIATE FLIGHT-LINE REMEDIES:**
- **ELECTRIC FUEL PUMP:** Switch ON immediately to clear fuel line bubbles.
- **THROTTLE POSITION:** Advance manually to **1400–1500 RPM** to pull fresh, cool fuel from the airframe tank into the hot engine compartment.
- **CARBURETOR HEAT:** Ensure Carb Heat is **OFF** completely during taxi to avoid adding hot air to an already heat-soaked intake circuit.
"""
    },
    "DUAL LANE ELECTRICAL DIAGNOSTICS": {
        "reasoning_points": [
            "The Rotax fuel-injected 'iS' engines utilize an internal permanent magnet generator supplying independent electrical networks: Lane A and Lane B via Generator A (Internal power) and Generator B (External/Battery charge power).",
            "A voltage drop below 12.0 Volts on either Lane indicates a potential regulator failure, stator winding degradation, bad grounding, or a faulty backup battery cross-feed circuit.",
            "Lane checks must be executed strictly following the cockpit Lane Selector protocol to isolate the EMS (Engine Management System) power source safely without stalling the fuel pumps."
        ],
        "specs_and_tooling_markdown": """
- **Engine System Type:** Fuel Injected EMS System (912iS / 915iS / 916iS architectures).
- **Required Diagnostic Tooling:** Calibrated digital multimeter (DMM) and specialized wiring diagram schematic.

---
##### **Engine Electrical Operation Limits:**
- **Normal Operating Bus Voltage:** **13.5V to 14.2V** on both networks when engine RPM is above 2500.
- **Minimum Low-Voltage Limit:** **12.0V**. Below this threshold, the ECU may drop sensors or fail to trigger fuel injectors correctly.
- **Generator B Output Rating:** Nominal 12V DC, max 30A continuous capacity.

---
##### **CRITICAL CHECKPOINTS:**
- **LANE CHECK PROTOCOL:** Turn Lane switch OFF only at recommended test RPM (typically 2000 RPM). Ensure opposite Lane remains stable and engine does not stumble.
- **GROUND BUS:** Inspect the main engine grounding strap. High impedance here causes immediate asymmetric lane voltage drops.
"""
    }
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
# 7. OPENROUTER HANDSHAKE WITH INJECTABLE SYSTEM PROMPT
# =====================================================
def call_llm(system_instructions: str, user_context: str):
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "meta-llama/llama-3.1-8b-instruct",
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": user_context}
        ],
        "providers": {"order": ["Lepton", "Together"], "allow_fallbacks": True}
    }
    response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
    response_json = response.json()

    if "error" in response_json:
        raise Exception(f"LLM API Error: {response_json['error']}")
    if not response_json.get("choices") or not response_json["choices"][0].get("message"):
        raise Exception(f"Invalid LLM response structure: {response_json}")

    return response_json["choices"][0]["message"]["content"]

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
# 10. USER COMMAND RUNNER WITH MULTI-TOPIC ROUTER
# =====================================================
user_query = st.chat_input("Enter technical maintenance question...")

if user_query:
    current_time = time.time()
    time_passed = current_time - st.session_state.last_query_time
    col_ctx = st.container() if is_admin_mode else center_console
    with col_ctx:
        with st.chat_message("user"): st.write(user_query)

    # ACTIVE SYSTEM REJECTION & TECHNICAL ESCALATION INTERCEPT CODES
    if st.session_state.active_engine is None:
        engine_match = re.search(r'(912\s*uls|912\s*ul|912\s*is|914|915\s*is|915|916\s*is|916)', user_query.lower())
        if engine_match:
            st.session_state.active_engine = engine_match.group(1).upper().replace(" ", "")
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.session_state.messages.append({"role": "assistant", "content": f"### 🔓 WORKSPACE UNLOCKED\nEngine profile securely set to **ROTAX {st.session_state.active_engine}**."})
            st.rerun()
        else:
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "⚠️ **ENGINE PROFILE CONFIGURATION REQUIRED**\n\nI cannot diagnose telemetry or look up specifications until your exact engine model is locked in. Lane architecture and tolerances vary significantly between carburetor assemblies and fuel-injected EMS blocks.\n\n**Please specify your exact engine model to unlock the workbench:**\n* **912UL** | **912ULS** | **912iS** | **914** | **915iS** | **916iS**"
            })
            st.rerun()

    # HARDENED DUAL-INTELLIGENCE TOPIC STATE ROUTER WITH VAPOR LOCK DETECTION
    if any(w in user_query.lower() for w in ["drop", "taxi", "heat", "boil", "soak", "lock", "800", "900"]):
        st.session_state.active_topic = "VAPOR LOCK AND HEAT SOAK DIAGNOSTICS"
    elif any(w in user_query.lower() for w in ["lane", "volt", "efis", "bus", "generator", "stator"]):
        st.session_state.active_topic = "DUAL LANE ELECTRICAL DIAGNOSTICS"
    elif any(w in user_query.lower() for w in ["carb", "sync", "balance", "float", "choke"]):
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
            st.error("Incompatible component configuration for this engine profile.")
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
                    
                    topic_data = SPEC_REGISTRY.get(st.session_state.active_topic)
                    if topic_data:
                        reasoning_points = "\n".join([f"- {point}" for point in topic_data["reasoning_points"]])
                        specs_markdown = topic_data["specs_and_tooling_markdown"]
                    else:
                        reasoning_points = "Focus on safety and technical accuracy."
                        specs_markdown = "Refer to official technical aviation manuals limits."

                    system_instructions = f"""You are 'Otimo Inspector', an expert AI mentor for aerospace technicians working on ROTAX engines. You are precise, highly technical, and safety-focused. Your job is to address the technician's actual maintenance issues clearly using the verified technical data provided.

You MUST structure your response using this exact three-part format:

### 1. THE WORKBENCH PROCEDURE
- Provide a clear, step-by-step mechanical walkthrough to address the technician's query.
- Use the 'MANDATORY REASONING POINTS' provided below to explain the engineering reason behind critical steps.
- You may incorporate matching contextual details from the 'REFERENCE EXTRACTS', but the Mandatory Points and Specifications must always take absolute priority.
- If the technician switches context to ask an adjacent question (e.g., asking about electrical lanes or gauge warnings mid-procedure), do not ignore it or force them back to a previous topic. Answer the active query step-by-step using the active data context provided.
- **CRITICAL INLINE SAFETY GATES:** If a step involves danger or high risk (such as working around live electrical buses, spinning propeller arcs, or systems under fluid pressure), you MUST call out that danger explicitly *at that exact step*. Immediately add a mandatory prompt instructing the user: "If you lack the confidence or specialized tools to proceed with this activity—as errors here may cause critical mechanical failure, severe personal harm, or death—STOP WORK immediately and contact a certified iRMT inspector."

### 2. ⚠️ INSPECTOR'S SAFETY BRIEF
- Highlight the 2-3 most critical, high-risk failure modes or mechanical blunders specific to this active task.
- Emphasize what can go wrong if specifications are ignored.
- **MANDATORY ESCALATION CLOSURE:** Conclude this section by advising the user that if they lack the confidence or specialized tools for any step, they must step back and contact a qualified iRMT technician.

### 3. REQUIRED SPECS & TOOLING
- Copy the text from the 'MANDATORY SPECIFICATIONS MARKDOWN' block provided in the user context exactly, 1:1, as a clean markdown list. Do not alter the numbers, units, or constraints.

GENERAL COMPLIANCE RULES:
- **FORMATTING:** Always ensure there is a clear blank line and three hashtags (###) before every major section header so the layout renders correctly on the workbench screen.
- Focus entirely on the active maintenance topic. Do not pull in data from other unrelated tasks.
- Do not mention, discuss, or provide instructions for two-stroke (2-stroke) engines or non-ROTAX systems.
"""

                    user_context = f"""---
MANDATORY REASONING POINTS FOR: {st.session_state.active_topic or 'General Inquiry'}
{reasoning_points}
---
MANDATORY SPECIFICATIONS MARKDOWN FOR: {st.session_state.active_topic or 'General Inquiry'}
(COPY THIS BLOCK EXACTLY INTO THE "REQUIRED SPECS & TOOLING" SECTION)
{specs_markdown}
---
TECHNICIAN'S QUERY: "{user_query}"
REFERENCE EXTRACTS: {context_str}
ENGINE: ROTAX {st.session_state.active_engine}
---
"""
                    assistant_response = call_llm(system_instructions, user_context)
                    st.session_state.daily_token_consumption += len(system_instructions.split()) + len(user_context.split()) + 1500
                    
                    if citations_map:
                        footer = "\n\n---\n\n### 📄 KEY MANUAL REFERENCES\n"
                        for doc, pages in citations_map.items():
                            footer += f"* **{doc}** — Page(s): {', '.join(map(str, sorted(list(pages))))}\n"
                        assistant_response += footer

                    response_placeholder.write(assistant_response)
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                    st.rerun()
                except Exception as e: 
                    st.error(f"An error occurred while generating the response: {str(e)}")
                    st.stop()