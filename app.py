import os
import streamlit as st
import re
import json
import hashlib
import time
import numpy as np
import requests
import faiss
from pypdf import PdfReader
from openai import OpenAI

# =====================================================
# 1. PAGE CONFIGURATION & INJECTED STRUCTURAL CSS
# =====================================================
is_admin_mode = (st.query_params.get("admin") == "true")

st.set_page_config(
    page_title="Otimo Aero AI Technician", 
    page_icon="✈️", 
    layout="wide",
    initial_sidebar_state="expanded" if is_admin_mode else "collapsed"
)

# Global stylesheet to constrain chat elements and programmatically override toggle handles
st.markdown("""
    <style>
    div[data-testid="stChatInput"] { max-width: 70% !important; margin: 0 auto !important; }
    .stChatInputContainer { max-width: 70% !important; margin: 0 auto !important; }
    .block-container { padding-bottom: 150px !important; } 
    """ + ("" if is_admin_mode else '[data-testid="stSidebar"] { display: none !important; }') + """
    </style>
    """, unsafe_allow_html=True)

# =====================================================
# 2. API CONFIGURATION & SAFETY GATES
# =====================================================
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")

if not OPENROUTER_API_KEY or not OPENAI_API_KEY:
    st.error("Missing required API credentials in Streamlit Secrets.")
    st.stop()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
openai_client = OpenAI(api_key=OPENAI_API_KEY)

INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "faiss_metadata.json"
CACHE_PATH = "embedding_cache.json"

# =====================================================
# 3. DYNAMIC MASTER SPEC REGISTRY
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
# 4. INITIALIZATION & STORAGE CACHE (OPTIMIZATION)
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

# =================================