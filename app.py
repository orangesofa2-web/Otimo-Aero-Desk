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
DAILY_TOKEN_BUDGET = 1100000 

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
- **Refill Quantity:** Approx. 3.0 Litres baseline capacity (final level MUST be verified via dipstick after expansion purging protocol).

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
"""
    },
    "SCHEDULED 100HR / 200HR INSPECTION": {
        "reasoning_points": [
            "The 100-hour and 200-hour tracking blocks represent the foundational airworthiness checks for both carburetted and fuel-injected Rotax lines.",
            "For fuel-injected engines (912iS, 915iS, 916iS), extracting the diagnostic ECU logs via BUDS software is mandatory to identify hidden sensor faults, knock alerts, or wastegate position errors.",
            "Inspecting the exhaust wastegate linkage on turbocharged variants (914, 915iS, 916iS) prevents severe overboost conditions or total manifold pressure failures during flight."
        ],
        "specs_and_tooling_markdown": """
- **Core Action Requirement:** Complete oil change, filter replacement, and magnetic plug extraction.
- **ECU Diagnostics (iS Variant Mandatory Metric):** Connect BUDS hardware tool and perform full error log dump.
- **Turbo System Checks (914 / 915iS / 916iS Mandatory Metric):** Check wastegate linkage free-movement clearance. Apply heat-resistant lubricant to the actuator joints.

---
##### **Fluid Capacities Matrix (Official Benchmarks):**
- **Rotax 912 UL / ULS / 914:** Approx. 3.0 Litres baseline capacity.
- **Rotax 912iS / 915iS / 916iS (Dry Sump Network):** Approx. 3.0 Litres baseline tank capacity (Total expansion network holds 3.2 to 3.4 Litres max capacity depending on airframe hoses).
"""
    },
    "GENERAL MAINTENANCE INQUIRY": {
        "reasoning_points": [
            "Aviation troubleshooting requires structural discipline. Conceptual mechanics can be discussed, but hard numbers must originate exclusively from official type-conforming documents.",
            "Unmapped operations must prioritize general system functionality and safe workspace practice over auto-completing numbers or specs."
        ],
        "specs_and_tooling_markdown": """
- **Core Directive:** Conceptual walkthrough only.
- **SPECIFICATION ACQUISITION:** You are required to cross-reference and extract exact numeric specifications, torque limits, and part clearances directly from your official hardcopy Rotax Line Maintenance Manual or airframe handbook. Do not rely on unverified memory.
"""
    }
}

# =====================================================
# 4. SESSION STATE INITIALIZATION & DISCLAIMERS
# =====================================================
if "documents" not in st.session_state: st.session_state.documents = []
if "last_query_time" not in st.session_state: st.session_state.last_query_time = 0.0
if "daily_token_consumption" not in st.session_state: st.session_state.daily_token_consumption = 0
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

# =====================================================
# 5. CORE FUNCTIONS (EMBEDDINGS & LLM HANDSHAKE)
# =====================================================
def invalid_configuration(query: str, engine_profile: str = None) -> bool:
    q = query.lower().replace(" ", "").replace("-", "")
    carb_terms = ["carb", "sync", "balance", "float", "choke"]
    injected_engines = ["915is", "916is", "912is"]
    return any(t in q for t in carb_terms) and (any(e in q for e in injected_engines) or any(e in (engine_profile or "").lower() for e in injected_engines))

def get_embedding(text: str, model="text-embedding-3-small"):
    return openai_client.embeddings.create(input=[text.replace("\n", " ")], model=model).data[0].embedding

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
    return response_json["choices"][0]["message"]["content"]

# =====================================================
# 6. WORKSPACE RENDERER
# =====================================================
def render_main_workspace():
    st.title("Otimo Aero AI Technician")
    status_line = f"Workspace Status — Engine Profile: {st.session_state.active_engine or 'NOT INITIALISED'}"
    if st.session_state.active_topic: status_line += f" | Task: {st.session_state.active_topic}"
    st.subheader(status_line)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.write(message["content"])

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
# 7. USER COMMAND RUNNER & TOPIC ROUTER
# =====================================================
_, center_console, _ = st.columns([0.15, 0.70, 0.15])
with center_console: render_main_workspace()

user_query = st.chat_input("Enter technical maintenance question...")

if user_query:
    current_time = time.time()
    if st.session_state.active_engine is None:
        engine_match = re.search(r'(912\s*uls|912\s*ul|912\s*is|914|915\s*is|915|916\s*is|916)', user_query.lower())
        if engine_match:
            raw_match = engine_match.group(1).upper().replace(" ", "")
            if raw_match == "915": raw_match = "915IS"
            if raw_match == "916": raw_match = "916IS"
            st.session_state.active_engine = raw_match
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.session_state.messages.append({"role": "assistant", "content": f"### 🔓 WORKSPACE UNLOCKED\nEngine profile securely set to **ROTAX {st.session_state.active_engine}**."})
            st.rerun()
        else:
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.session_state.messages.append({"role": "assistant", "content": "⚠️ **ENGINE PROFILE CONFIGURATION REQUIRED**\n\nPlease specify engine: **912UL | 912ULS | 912iS | 914 | 915iS | 916iS**"})
            st.rerun()

    # TOPIC ROUTER
    if any(w in user_query.lower() for w in ["service", "hour", "100", "200", "schedule", "interval"]): st.session_state.active_topic = "SCHEDULED 100HR / 200HR INSPECTION"
    elif any(w in user_query.lower() for w in ["drop", "taxi", "heat", "boil", "soak", "lock", "800", "900"]): st.session_state.active_topic = "VAPOR LOCK AND HEAT SOAK DIAGNOSTICS"
    elif any(w in user_query.lower() for w in ["lane", "volt", "efis", "bus", "generator", "stator"]): st.session_state.active_topic = "DUAL LANE ELECTRICAL DIAGNOSTICS"
    elif any(w in user_query.lower() for w in ["carb", "sync", "balance", "float", "choke"]): st.session_state.active_topic = "CARBURETOR SYNCHRONIZATION"
    elif any(w in user_query.lower() for w in ["plug", "gap", "spark"]): st.session_state.active_topic = "SPARK PLUG INSPECTION"
    elif any(w in user_query.lower() for w in ["pressure", "gauge"]) and "oil" in user_query.lower(): st.session_state.active_topic = "OIL PRESSURE CHECK"
    elif any(w in user_query.lower() for w in ["drain", "magnet", "change", "oil"]): st.session_state.active_topic = "OIL CHANGE / MAGNETIC PLUG INSPECTION"
    else: st.session_state.active_topic = "GENERAL MAINTENANCE INQUIRY"

    st.session_state.messages.append({"role": "user", "content": user_query})

    with center_console.chat_message("assistant"):
        if invalid_configuration(user_query, st.session_state.active_engine):
            st.error("Incompatible component configuration for this engine profile.")
            st.stop()
        else:
            with st.spinner("Executing spatial context scan..."):
                try:
                    search_query = f"{st.session_state.active_engine} {st.session_state.active_topic or ''} {user_query}"
                    context_str = "No specific match found."
                    citations = {}
                    
                    if st.session_state.vector_index is not None:
                        query_vector = np.array([get_embedding(search_query)]).astype('float32')
                        dist, ind = st.session_state.vector_index.search(query_vector, 3)
                        chunks = [st.session_state.vector_metadata[i]['text'] for i in ind[0] if i != -1]
                        context_str = "\n\n---\n\n".join(chunks)

                    topic_data = SPEC_REGISTRY.get(st.session_state.active_topic)
                    reasoning_points = "\n".join([f"- {p}" for p in topic_data["reasoning_points"]]) if topic_data else ""
                    specs_markdown = topic_data["specs_and_tooling_markdown"] if topic_data else "Refer to hardcopy manual."

                    system_instructions = """You are 'Otimo Inspector', an expert aerospace AI mentor. Address maintenance tasks precisely. 
                    1. THE WORKBENCH PROCEDURE: Step-by-step walkthrough.
                    2. ⚠️ INSPECTOR'S SAFETY BRIEF: High-risk failure modes & mandatory iRMT escalation for lack of confidence.
                    3. REQUIRED SPECS & TOOLING: Copy markdown table strictly.
                    DO NOT invent numbers. If data is missing, state it must be verified in the hardcopy manual."""
                    
                    user_context = f"Topic: {st.session_state.active_topic}\n\nReasoning:\n{reasoning_points}\n\nSpecs:\n{specs_markdown}\n\nQuery: {user_query}"
                    
                    response = call_llm(system_instructions, user_context)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.rerun()
                except Exception as e:
                    st.error(f"System Error: {str(e)}")