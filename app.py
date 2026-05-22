import os
import streamlit as st

# Essential config
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
os.environ["STREAMLIT_SERVER_PORT"] = "8080"

# REMOVE the enforce_referral_source() call entirely.
# It is the single biggest risk factor right now.
# If you want to track analytics, do it in a way that doesn't 
# involve any conditional logic that can trigger st.stop().

import re
import json
import hashlib
import time
import numpy as np
import requests
import faiss
from pypdf import PdfReader
from openai import OpenAI

# ... rest of your code ...

# ... (Rest of your original application logic)

# ... Rest of your application ...

# ... rest of your code ...

def get_secret(key):
    # Aggressively prioritize Environment Variables first
    val = os.environ.get(key)
    if val:
        return val
    
    # Only if not in Cloud Run and not in Env, check st.secrets
    if "K_SERVICE" not in os.environ:
        try:
            import streamlit as st
            return st.secrets.get(key)
        except:
            return None
    return None

OPENROUTER_API_KEY = get_secret("OPENROUTER_API_KEY")
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
ADMIN_PASSWORD = get_secret("ADMIN_PASSWORD")

# =====================================================
# 1. PAGE CONFIGURATION & INJECTED STRUCTURAL CSS
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
# 2. DOMAIN GUARDRAIL & SECURITY ARCHITECTURE
# =====================================================
# CONFIGURATION: Set this to your actual website domain
ALLOWED_PARENT_DOMAIN = "otimoaero.com" 

def verify_hosting_environment():
    """Security Layer: Ensures the app is strictly framed inside your website."""
    # Streamlit headers expose the parent hosting context via context headers
    headers = st.context.headers
    referer = headers.get("Referer", "")
    ancestor = headers.get("Sec-Fetch-Dest", "")
    
    # In local development, allow localhost to pass through smoothly
    if "localhost" in referer or "127.0.0.1" in referer:
        return True
        
    # In production, check if your domain is present in the referring headers
    if ALLOWED_PARENT_DOMAIN not in referer:
        st.error("🔒 **Access Denied:** Direct access to this terminal engine is prohibited. Please access via the official portal.")
        st.stop()

# Trigger Domain Lock Check
# verify_hosting_environment()

# =====================================================
# 3. API CONFIGURATION & SAFETY GATES
# =====================================================
# We rely EXCLUSIVELY on environment variables. 
# st.secrets is ignored to prevent Streamlit SecretNotFoundError.

def get_secret(key):
    """Aggressively prioritizes Environment Variables only."""
    return os.environ.get(key)

OPENROUTER_API_KEY = get_secret("OPENROUTER_API_KEY")
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
ADMIN_PASSWORD = get_secret("ADMIN_PASSWORD")

# Hard gate for mandatory credentials
if not OPENROUTER_API_KEY or not OPENAI_API_KEY or not ADMIN_PASSWORD:
    st.error("Configuration Error: Required API keys or ADMIN_PASSWORD missing from Environment Variables.")
    st.stop()

INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "faiss_metadata.json"
CACHE_PATH = "embedding_cache.json"

# =====================================================
# 4. DYNAMIC MASTER SPEC REGISTRY
# =====================================================
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
# 5. INITIALIZATION & STORAGE CACHE
# =====================================================
if "active_engine" not in st.session_state: st.session_state.active_engine = None
if "active_topic" not in st.session_state: st.session_state.active_topic = None
if "documents" not in st.session_state: st.session_state.documents = []
if "messages" not in st.session_state: 
    st.session_state.messages = [{
        "role": "assistant", 
        "content": "### 🔧 Engine Selection Required\nWelcome to the workbench! Before we look up any technical maintenance details, we need to lock onto your precise engine configuration.\n\n**🚨 IMPORTANT MAINTENANCE DIRECTIVE / TECHNICAL DISCLAIMER 🚨**\n*This AI system is highly experimental and serves strictly as an informational guide. All users must cross-reference and double-check instructions, tolerances, and part arrays against official hardcopy documentation before altering any flight system. If in any doubt regarding configuration safety, immediately stop work and contact a qualified iRMT.*\n\n**Please reply with the specific engine type you are working on today:**\n* **912UL** | **912ULS** | **912iS** | **914** | **915iS** | **916iS**"
    }]

if "embed_cache" not in st.session_state:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f: st.session_state.embed_cache = json.load(f)
    else: st.session_state.embed_cache = {}

if "vector_index" not in st.session_state:
    if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
        try:
            st.session_state.vector_index = faiss.read_index(INDEX_PATH)
            with open(METADATA_PATH, "r", encoding="utf-8") as f: st.session_state.vector_metadata = json.load(f)
            st.session_state.documents = list(set(m["source"] for m in st.session_state.vector_metadata))
        except Exception: st.session_state.vector_index, st.session_state.vector_metadata = None, []
    else: st.session_state.vector_index, st.session_state.vector_metadata = None, []

# =====================================================
# 6. EMBEDDING PIPELINE & VECTOR FUNCTIONS
# =====================================================
def get_embeddings_batched(texts, model="text-embedding-3-small"):
    results = [None] * len(texts)
    uncached_texts, uncached_indices = [], []

    for i, text in enumerate(texts):
        chunk_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        if chunk_hash in st.session_state.embed_cache: results[i] = st.session_state.embed_cache[chunk_hash]
        else:
            uncached_texts.append(text)
            uncached_indices.append(i)

    if uncached_texts:
        BATCH_SIZE = 100
        new_embeddings = []
        for i in range(0, len(uncached_texts), BATCH_SIZE):
            batch = [t.replace("\n", " ") for t in uncached_texts[i:i+BATCH_SIZE]]
            response = openai_client.embeddings.create(input=batch, model=model)
            new_embeddings.extend([d.embedding for d in response.data])
            time.sleep(0.05)

        for i, text in enumerate(uncached_texts):
            chunk_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
            st.session_state.embed_cache[chunk_hash] = new_embeddings[i]
            results[uncached_indices[i]] = new_embeddings[i]

        with open(CACHE_PATH, "w", encoding="utf-8") as f: json.dump(st.session_state.embed_cache, f)
    return results

def invalid_configuration(query: str, engine_profile: str = None) -> bool:
    q = query.lower().replace(" ", "").replace("-", "")
    carb_terms = ["carb", "sync", "balance", "float", "choke"]
    injected_engines = ["915", "916", "912is"]
    return any(t in q for t in carb_terms) and (any(e in q for e in injected_engines) or any(e in (engine_profile or "").lower() for e in injected_engines))

def parse_and_chunk_pdf(uploaded_files):
    all_chunks = []
    for uploaded_file in uploaded_files:
        try:
            reader = PdfReader(uploaded_file)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if not text: continue
                normalized_text = re.sub(r'\s+', ' ', text).strip()
                sentences = re.split(r'(?<=[.!?])\s+', normalized_text)
                current_chunk, current_word_count = [], 0
                
                for sentence in sentences:
                    sentence_words = sentence.split()
                    if current_word_count + len(sentence_words) > 120:
                        chunk_str = " ".join(current_chunk)
                        if len(chunk_str.strip()) > 40:
                            all_chunks.append({"text": chunk_str, "source": uploaded_file.name, "page": page_num + 1})
                        current_chunk = current_chunk[-3:] if len(current_chunk) > 3 else current_chunk
                        current_word_count = sum(len(s.split()) for s in current_chunk)
                    current_chunk.append(sentence)
                    current_word_count += len(sentence_words)
                if current_chunk:
                    chunk_str = " ".join(current_chunk)
                    if len(chunk_str.strip()) > 40:
                        all_chunks.append({"text": chunk_str, "source": uploaded_file.name, "page": page_num + 1})
        except Exception as e: st.error(f"Error parsing text streams: {str(e)}")
            
    if all_chunks:
        with st.spinner("Processing optimization tokens into secure cache layer..."):
            texts = [c["text"] for c in all_chunks]
            embeddings = get_embeddings_batched(texts)
            if embeddings:
                embeddings_array = np.array(embeddings).astype('float32')
                faiss.normalize_L2(embeddings_array)
                index = faiss.IndexFlatIP(len(embeddings_array[0]))
                index.add(embeddings_array)
                faiss.write_index(index, INDEX_PATH)
                with open(METADATA_PATH, "w", encoding="utf-8") as f: json.dump(all_chunks, f, ensure_ascii=False, indent=2)
                st.success("Universal localized system vector database synchronized!")
                st.rerun()

# =====================================================
# 7. PROMPT ENGINEERING SYSTEM HUB
# =====================================================
BASE_SYSTEM_PROMPT = """You are 'Otimo Inspector', an expert AI mentor for aerospace technicians working on ROTAX engines. You are precise, highly technical, and safety-focused. Your job is to address the technician's actual maintenance issues clearly using the verified technical data provided.

You MUST structure your response using this exact three-part format:

### 1. THE WORKBENCH PROCEDURE
- Provide a clear, step-by-step mechanical walkthrough to address the technician's query.
- Use the 'MANDATORY REASONING POINTS' to explain the engineering reason behind critical steps.
- **CRITICAL INLINE SAFETY GATES:** If a step involves danger or high risk, call it out explicitly *at that exact step*. Add: "If you lack the confidence or specialized tools to proceed with this activity—as errors here may cause critical mechanical failure, severe personal harm, or death—STOP WORK immediately and contact a certified iRMT inspector."

### 2. ⚠️ INSPECTOR'S SAFETY BRIEF
- Highlight the 2-3 most critical, high-risk failure modes specific to this active task.
- **MANDATORY ESCALATION CLOSURE:** Conclude exactly with: "If you lack the confidence or specialized tools for any step, you must step back and contact a qualified iRMT technician."

### 3. REQUIRED SPECS & TOOLING
- Copy the text from the 'MANDATORY SPECIFICATIONS MARKDOWN' block provided in the user context exactly, 1:1, as a clean markdown list. Do not alter numbers.

STRICT DISCIPLINE RULES:
- **IDENTITY:** You are an AI model, NOT an iRMT inspector. Never refer to yourself as an inspector.
- **CARBURETOR HALLUCINATION BAN:** Fuel injected architectures (912iS, 915iS, 916iS) possess NO chokes, float bowls, or mixture screws. Completely reject any context fragments matching carburetor settings if the active engine profile is an 'iS' variant.
- **GROUNDING ENFORCEMENT:** If the query is completely outside the scope of the provided specifications, state cleanly: "Verification profile data unavailable in loaded documentation references." """

def call_llm(user_context: str, chat_history: list):
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    api_messages = [{"role": "system", "content": BASE_SYSTEM_PROMPT}]
    pruned_history = chat_history[-4:] if len(chat_history) > 4 else chat_history
    for msg in pruned_history:
        if msg["content"] != user_context: api_messages.append({"role": msg["role"], "content": msg["content"]})
    api_messages.append({"role": "user", "content": user_context})

    payload = {
        "model": "meta-llama/llama-3.1-8b-instruct",
        "temperature": 0.1,
        "messages": api_messages,
        "providers": {"order": ["Lepton", "Together"], "allow_fallbacks": True}
    }
    response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=90)
    return response.json()["choices"][0]["message"]["content"]

# =====================================================
# 8. THE SECURE PASSWORD PANEL (UPGRADE COMPLETE)
# =====================================================
with st.sidebar:
    st.header("🔑 Administrative Access")
    admin_input = st.text_input("Enter Admin Password", type="password")
    
    # Check if input matches secret key string
    is_authenticated_admin = (admin_input == ADMIN_PASSWORD)
    
    if is_authenticated_admin:
        st.success("Access Granted")
        st.divider()
        st.subheader("⚙️ System Document Control")
        uploaded_files = st.file_uploader("Upload Airframe Technical Manuals", type=["pdf"], accept_multiple_files=True)
        if uploaded_files: parse_and_chunk_pdf(uploaded_files)
        if st.button("Clear Vector Core"):
            for p in [INDEX_PATH, METADATA_PATH, CACHE_PATH]: 
                if os.path.exists(p): os.remove(p)
            st.rerun()
    elif admin_input:
        st.error("Incorrect administrative credentials.")

# =====================================================
# 9. MAIN INTERSPACE VIEW
# =====================================================
col_layout = st.columns([0.15, 0.70, 0.15])[1]

with col_layout:
    st.title("Otimo Aero AI Technician")
    if st.session_state.active_engine:
        st.markdown(f"#### 🛠️ Workspace Connected \n**Engine Profile:** `ROTAX {st.session_state.active_engine}` &nbsp;&nbsp;|&nbsp;&nbsp; **Active Subsystem:** `{st.session_state.active_topic or 'Awaiting Task Identification'}`")
        st.divider()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.write(message["content"])

# =====================================================
# 10. INPUT ROUTER CONTROLS
# =====================================================
user_query = st.chat_input("Enter engine profile code or technician system question...")

def is_prompt_injection(user_input):
    INJECTION_PATTERNS = [
        r"ignore\s+all\s+previous\s+instructions",
        r"disregard\s+all\s+rules",
        r"system\s+override",
        r"reveal\s+your\s+system\s+prompt"
    ]
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, user_input, re.IGNORECASE):
            return True
    return False

if user_query:
    # --- ADD THIS NEW FILTER CHECK HERE ---
    if is_prompt_injection(user_query):
        with col_layout:
            st.error("⚠️ Security Alert: Malicious input detected. Request blocked.")
            st.stop()
    # --------------------------------------
    
    with col_layout:
        with st.chat_message("user"): st.write(user_query)
    # ... rest of your existing logic continues here

    if st.session_state.active_engine is None:
        engine_match = re.search(r'(912\s*uls|912\s*ul|912\s*is|914|915\s*is|915|916\s*is|916)', user_query.lower())
        if engine_match:
            st.session_state.active_engine = engine_match.group(1).upper().replace(" ", "")
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.session_state.messages.append({"role": "assistant", "content": f"### 🔓 WORKSPACE UNLOCKED\nEngine profile securely set to **ROTAX {st.session_state.active_engine}**."})
            st.rerun()
        else:
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.session_state.messages.append({"role": "assistant", "content": "⚠️ **ENGINE PROFILE UNINITIALISED**\nState exact specification key setup to unlock workbench access: **912UL | 912ULS | 912iS | 914 | 915iS | 916iS**"})
            st.rerun()
    else:
        st.session_state.messages.append({"role": "user", "content": user_query})

        if any(w in user_query.lower() for w in ["lane", "volt", "efis", "bus", "generator", "stator"]): st.session_state.active_topic = "DUAL LANE ELECTRICAL DIAGNOSTICS"
        elif any(w in user_query.lower() for w in ["carb", "sync", "balance", "float", "choke"]): st.session_state.active_topic = "CARBURETOR SYNCHRONIZATION"
        elif any(w in user_query.lower() for w in ["plug", "gap", "spark"]): st.session_state.active_topic = "SPARK PLUG INSPECTION"
        elif any(w in user_query.lower() for w in ["pressure", "gauge"]) and "oil" in user_query.lower(): st.session_state.active_topic = "OIL PRESSURE CHECK"
        elif any(w in user_query.lower() for w in ["drain", "magnet", "change", "oil"]): st.session_state.active_topic = "OIL CHANGE / MAGNETIC PLUG INSPECTION"
        else: st.session_state.active_topic = "GENERAL MAINTENANCE INQUIRY"

        with col_layout.chat_message("assistant"):
            if invalid_configuration(user_query, st.session_state.active_engine):
                st.error("Critical airworthiness conflict: Attempting carburetor operations on an integrated fuel-injected EMS architecture block. Process terminated.")
                st.stop()
            else:
                with st.spinner("Scanning vectorized indices for verified technical cross-references..."):
                    try:
                        context_str, citations_map = "No matching manual data extracted.", {}
                        search_query = f"ROTAX {st.session_state.active_engine} {st.session_state.active_topic} {user_query}"

                        if st.session_state.vector_index is not None:
                            query_vector = np.array([get_embeddings_batched([search_query])[0]]).astype('float32')
                            faiss.normalize_L2(query_vector)
                            distances, indices = st.session_state.vector_index.search(query_vector, 5)
                            matched_chunks = []
                            for score, idx in zip(distances[0], indices[0]):
                                if idx != -1 and score > 0.55 and idx < len(st.session_state.vector_metadata):
                                    chunk_data = st.session_state.vector_metadata[idx]
                                    matched_chunks.append(chunk_data['text'])
                                    citations_map.setdefault(chunk_data['source'], set()).add(chunk_data['page'])
                            if matched_chunks: context_str = "\n\n---\n\n".join(matched_chunks)
                        
                        topic_data = SPEC_REGISTRY.get(st.session_state.active_topic)
                        reasoning_points = "\n".join([f"- {point}" for point in topic_data["reasoning_points"]]) if topic_data else "Verify maintenance alignment."
                        specs_markdown = topic_data["specs_and_tooling_markdown"] if topic_data else "No lookup values configured."

                        user_context = f"""---
MANDATORY REASONING POINTS FOR: {st.session_state.active_topic}
{reasoning_points}
---
MANDATORY SPECIFICATIONS MARKDOWN FOR: {st.session_state.active_topic}
{specs_markdown}
---
ENGINE MODEL IDENTIFICATION: ROTAX {st.session_state.active_engine}
REFERENCE EXTRACTS FROM LOADED DOCUMENTS:
{context_str}
"""
                        assistant_response = call_llm(user_context, st.session_state.messages)
                        if citations_map:
                            footer = "\n\n---\n\n### 📄 KEY MANUAL REFERENCES\n"
                            for doc, pages in citations_map.items():
                                footer += f"* **{doc}** — Page(s): {', '.join(map(str, sorted(list(pages))))}\n"
                            assistant_response += footer

                        st.write(assistant_response)
                        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                    except Exception as e: st.error(f"Airworthiness processor pipeline failure: {str(e)}")