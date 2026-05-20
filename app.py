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

# Robust selector overrides targeting inner and outer framework components of st.chat_input
st.markdown(
    """
    <style>
    /* Match the bottom input container strictly to the 70% screen console region */
    div[data-testid="stChatInput"] {
        max-width: 70% !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }
    
    /* Ensure the sticky fixed-dock background matches alignment layout bounds */
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
    st.error("Missing required API credentials in Streamlit Secrets. Ensure both OPENROUTER_API_KEY and OPENAI_API_KEY are configured.")
    st.stop()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Initialize OpenAI Client strictly for high-dimensional semantic embeddings
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Vector File Paths for Persistent Storage on the Server
INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "faiss_metadata.json"

# =====================================================
# 3. SAFETY GUARDRAIL PARAMETERS (UPDATED BENCHMARKS)
# =====================================================
COOLDOWN_SECONDS = 5
MAX_QUERY_CHARACTERS = 400
DAILY_TOKEN_BUDGET = 450000

# =====================================================
# 4. SESSION STATE INITIALIZATION
# =====================================================
if "documents" not in st.session_state:
    st.session_state.documents = []

if "last_query_time" not in st.session_state:
    st.session_state.last_query_time = 0.0

if "daily_token_consumption" not in st.session_state:
    st.session_state.daily_token_consumption = 0

if "alert_triggered_today" not in st.session_state:
    st.session_state.alert_triggered_today = False

if "active_engine" not in st.session_state:
    st.session_state.active_engine = None

if "active_topic" not in st.session_state:
    st.session_state.active_topic = None

if "vector_index" not in st.session_state:
    if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
        try:
            st.session_state.vector_index = faiss.read_index(INDEX_PATH)
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                st.session_state.vector_metadata = json.load(f)
            st.session_state.documents = list(set(m["source"] for m in st.session_state.vector_metadata))
        except Exception:
            st.session_state.vector_index = None
            st.session_state.vector_metadata = []
    else:
        st.session_state.vector_index = None
        st.session_state.vector_metadata = []

WELCOME_PROMPT = """### 🔧 Engine Selection Required

Welcome to the workbench! Before we look up any technical maintenance details, we need to lock onto your precise engine configuration. 

Critical parameters—such as plug gaps, line-purging steps, fuel pressures, and torque values—vary significantly across model variants. Setting this filter ensures the search engine safely targets the correct technical manual documentation.

**Please reply with the specific engine type you are working on today:**
* **912UL**
* **912ULS**
* **912iS**
* **914**
* **915iS**
* **916iS**

*Type your matching engine key code below to open the maintenance desk channels.*"""

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": WELCOME_PROMPT}]

if "pending_clarification" not in st.session_state:
    st.session_state.pending_clarification = None

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
    
    is_carb_query = any(t in q for t in carb_terms)
    
    is_profile_injected = False
    if engine_profile:
        ep = engine_profile.lower().replace(" ", "").replace("-", "")
        is_profile_injected = any(e in ep for e in injected_engines)
        
    is_query_injected = any(e in q for e in injected_engines)
    
    return is_carb_query and (is_query_injected or is_profile_injected)

def get_embedding(text: str, model="text-embedding-3-small"):
    cleaned_text = text.replace("\n", " ")
    response = openai_client.embeddings.create(input=[cleaned_text], model=model)
    return response.data[0].embedding

def send_pushover_alert(title: str, message: str):
    if not PUSHOVER_USER_KEY or not PUSHOVER_APP_TOKEN:
        return
    try:
        url = "https://api.pushover.net/1/messages.json"
        payload = {
            "token": PUSHOVER_APP_TOKEN,
            "user": PUSHOVER_USER_KEY,
            "title": title,
            "message": message,
            "priority": 1,
            "sound": "siren"
        }
        requests.post(url, data=payload, timeout=10)
    except Exception:
        pass

# =====================================================
# 6. DOCUMENT INGESTION & VECTOR MATRIX BUILDER
# =====================================================
def rebuild_vector_database(uploaded_files):
    all_chunks = []
    for uploaded_file in uploaded_files:
        try:
            reader = PdfReader(uploaded_file)
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text and len(page_text.strip()) > 50:
                    all_chunks.append({
                        "text": page_text,
                        "source": uploaded_file.name,
                        "page": page_num + 1
                    })
        except Exception as e:
            st.error(f"Error parsing {uploaded_file.name}: {str(e)}")
            
    if all_chunks:
        embeddings_list = []
        metadata_list = []
        progress_bar = st.progress(0)
        st.write(f"Vectorising {len(all_chunks)} manual pages via OpenAI API...")
        
        for idx, chunk in enumerate(all_chunks):
            try:
                vec = get_embedding(chunk["text"])
                embeddings_list.append(vec)
                metadata_list.append(chunk)
            except Exception as api_err:
                st.error(f"Embedding failure on page unit {idx}: {str(api_err)}")
            progress_bar.progress((idx + 1) / len(all_chunks))
            
        if embeddings_list:
            dimension = len(embeddings_list[0])
            np_embeddings = np.array(embeddings_list).astype('float32')
            
            index = faiss.IndexFlatL2(dimension)
            index.add(np_embeddings)
            
            faiss.write_index(index, INDEX_PATH)
            with open(METADATA_PATH, "w", encoding="utf-8") as f:
                json.dump(metadata_list, f, ensure_ascii=False, indent=2)
                
            st.session_state.vector_index = index
            st.session_state.vector_metadata = metadata_list
            st.session_state.documents = list(set(m["source"] for m in metadata_list))
            st.success("Universal semantic database built and stored successfully!")
            st.rerun()

# =====================================================
# 7. OPENROUTER PRODUCTION HANDSHAKE (LLAMA 3.1 8B)
# =====================================================
def call_llm(prompt: str):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "meta-llama/llama-3.1-8b-instruct",
        "temperature": 0.0,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are the lead technical AI desk assistant for Otimo Aero, providing maintenance support directly to technicians working on aircraft. "
                    "You output answers in a strict, professional, itemized layout. No conversational fluff, meta-references, or unhelpful remarks."
                )
            },
            {"role": "user", "content": prompt}
        ],
        "providers": {
            "order": ["Lepton", "Together"],
            "allow_fallbacks": True
        }
    }
    
    response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
    if response.status_code != 200:
        raise Exception(response.text)
    return response.json()["choices"][0]["message"]["content"]

# =====================================================
# 8. SIDEBAR CONTROL PANEL (ADMIN GATE LOCKED)
# =====================================================
url_params = st.query_params
is_admin_mode = url_params.get("admin") == "true"

if is_admin_mode:
    with st.sidebar:
        st.header("⚙️ Admin Control Panel")
        if not st.session_state.documents:
            uploaded_files = st.file_uploader(
                "Upload Technical Manuals",
                type=["pdf"],
                accept_multiple_files=True
            )
            if uploaded_files:
                with st.spinner("Executing high-dimensional conceptual indexing..."):
                    rebuild_vector_database(uploaded_files)
        else:
            st.success("🔒 System Manuals Locked into Memory")
            if st.button("Clear & Re-upload Manuals"):
                if os.path.exists(INDEX_PATH): os.remove(INDEX_PATH)
                if os.path.exists(METADATA_PATH): os.remove(METADATA_PATH)
                st.session_state.vector_index = None
                st.session_state.vector_metadata = []
                st.session_state.documents = []
                st.session_state.active_topic = None
                st.rerun()

        st.divider()
        st.metric("Indexed Manuals", len(st.session_state.documents))
        st.metric("Searchable Vector Units", len(st.session_state.vector_metadata) if st.session_state.vector_metadata else 0)
        
        st.divider()
        st.subheader("Guardrail Budget Tracking")
        st.progress(min(st.session_state.daily_token_consumption / DAILY_TOKEN_BUDGET, 1.0))
        st.caption(f"Daily Token Counter: {st.session_state.daily_token_consumption} / {DAILY_TOKEN_BUDGET}")

# =====================================================
# 9. DISPLAY CONTENT GENERATOR MATRIX
# =====================================================
def render_main_workspace():
    st.title("Otimo Aero AI Technician")

    status_line = f"Workspace Status — Engine Profile: {st.session_state.active_engine if st.session_state.active_engine else 'NOT INITIALISED'}"
    if st.session_state.active_topic:
        status_line += f" | Current Maintenance Task: {st.session_state.active_topic}"
    st.subheader(status_line)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Render chat workspace history stack
if is_admin_mode:
    render_main_workspace()
else:
    left_margin, center_console, right_margin = st.columns([0.15, 0.70, 0.15])
    with center_console:
        render_main_workspace()

# =====================================================
# 10. USER COMMAND RUNNER WITH ARCHITECTURE HOOKS
# =====================================================
user_query = st.chat_input("Enter technical maintenance question...")

if user_query:
    current_time = time.time()
    time_passed = current_time - st.session_state.last_query_time
    
    # Process the user query layout printing step
    if is_admin_mode:
        with st.chat_message("user"):
            st.write(user_query)
    else:
        left_margin, center_console, right_margin = st.columns([0.15, 0.70, 0.15])
        with center_console:
            with st.chat_message("user"):
                st.write(user_query)

    # TEST TRIGGER
    if user_query.strip().upper() == "TEST_ALERT_NOW":
        st.session_state.daily_token_consumption = DAILY_TOKEN_BUDGET + 1000

    # ENGINE CONTEXT GATE
    if st.session_state.active_engine is None:
        engine_match = re.search(r'(912\s*uls|912\s*ul|912\s*is|914|915\s*is|915|916\s*is|916)', user_query.lower())
        if engine_match:
            normalized_engine = engine_match.group(1).upper().replace(" ", "")
            st.session_state.active_engine = normalized_engine
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"### 🔓 WORKSPACE UNLOCKED\nEngine profile securely set to **ROTAX {normalized_engine}**. Core data routing channels are now active. Please enter your primary technical maintenance query or task below."
            })
            st.rerun()
        else:
            if is_admin_mode:
                with st.chat_message("assistant"): st.markdown(WELCOME_PROMPT)
            else:
                left_margin, center_console, right_margin = st.columns([0.15, 0.70, 0.15])
                with center_console:
                    with st.chat_message("assistant"): st.markdown(WELCOME_PROMPT)
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.session_state.messages.append({"role": "assistant", "content": WELCOME_PROMPT})
            st.stop()

    # TOPIC EXTRACTOR LOGIC
    change_topic_match = re.search(r'(purge|oil|plug|spark|gap|torque|carb|balance|sync|pressure|fuel)', user_query.lower())
    if change_topic_match and "tool" not in user_query.lower():
        if "purge" in user_query.lower() or "oil" in user_query.lower():
            st.session_state.active_topic = "OIL PURGING"
        elif "plug" in user_query.lower() or "gap" in user_query.lower():
            st.session_state.active_topic = "SPARK PLUG INSPECTION"
        elif "carb" in user_query.lower() or "sync" in user_query.lower() or "balance" in user_query.lower():
            st.session_state.active_topic = "CARBURETOR SYNCHRONIZATION"
        elif "pressure" in user_query.lower() or "fuel" in user_query.lower():
            st.session_state.active_topic = "FUEL PRESSURE CHECK"

    # GUARDRAIL LAYER A: Cooldown Timer Enforcement
    if time_passed < COOLDOWN_SECONDS:
        wait_remainder = int(COOLDOWN_SECONDS - time_passed)
        error_msg = f"⏳ **RATE LIMIT TRIGGERED:** Please wait {wait_remainder} more seconds before submitting another question to protect system stability."
        if is_admin_mode:
            with st.chat_message("assistant"): st.warning(error_msg)
        else:
            left_margin, center_console, right_margin = st.columns([0.15, 0.70, 0.15])
            with center_console:
                with st.chat_message("assistant"): st.warning(error_msg)
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        st.stop()

    # GUARDRAIL LAYER B: Query Size Hard Cap
    if len(user_query) > MAX_QUERY_CHARACTERS:
        error_msg = f"⚠️ **INPUT OVERFLOW:** Your entry is too long ({len(user_query)} characters). Questions are limited to {MAX_QUERY_CHARACTERS} characters."
        if is_admin_mode:
            with st.chat_message("assistant"): st.error(error_msg)
        else:
            left_margin, center_console, right_margin = st.columns([0.15, 0.70, 0.15])
            with center_console:
                with st.chat_message("assistant"): st.error(error_msg)
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        st.stop()

    # GUARDRAIL LAYER C: Daily Token Budget Safety Brake
    if st.session_state.daily_token_consumption >= DAILY_TOKEN_BUDGET:
        if not st.session_state.alert_triggered_today:
            send_pushover_alert(
                title="🚨 Otimo Aero: Daily Budget Spent",
                message=f"The application has successfully hit its safety cap limit of {DAILY_TOKEN_BUDGET} tokens. Interface API traffic locked."
            )
            st.session_state.alert_triggered_today = True

        error_msg = "🚨 **EMERGENCY SHUTDOWN:** The application has reached its maximum daily data allotment. API requests have been locked down to prevent balance exhaustion. Please check again tomorrow."
        if is_admin_mode:
            with st.chat_message("assistant"): st.error(error_msg)
        else:
            left_margin, center_console, right_margin = st.columns([0.15, 0.70, 0.15])
            with center_console:
                with st.chat_message("assistant"): st.error(error_msg)
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        st.stop()

    # Record the timestamp of this verified query
    st.session_state.last_query_time = current_time
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Set assistant panel placement variables
    if is_admin_mode:
        assistant_canvas = st.chat_message("assistant")
    else:
        left_margin, center_console, right_margin = st.columns([0.15, 0.70, 0.15])
        assistant_canvas = center_console.chat_message("assistant")

    with assistant_canvas:
        response_placeholder = st.empty()

        # SCENARIO A: Resolving pending clarification requests
        if st.session_state.pending_clarification:
            original_intent = st.session_state.pending_clarification
            st.session_state.pending_clarification = None
            user_query = f"{original_intent} specifically regarding {user_query}"

        # SCENARIO B: Enforce specific engine variant selections
        if requires_variant(user_query):
            st.session_state.pending_clarification = user_query
            assistant_response = """### 🔍 SPECIFICATION REQUIRED
To provide the correct technical clearances or procedure parameters, please specify your exact engine model variant:
* **912ULS** (100 hp, Carbureted)
* **912UL** (80 hp, Carbureted)
* **912iS** (100 hp, Fuel Injected)

*Please type your variant directly into the chat input below to proceed.*"""
            response_placeholder.write(assistant_response)
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            st.stop()

        # SCENARIO C: Hardcoded Fuel Injection Component Gate
        if invalid_configuration(user_query, st.session_state.active_engine):
            assistant_response = """### 1. QUICK SPEC / PROCEDURE
* **CRITICAL ERROR:** The engine model specified configuration platform utilizes electronic fuel injection and does not possess carburetors.
* Carburetor synchronization and pneumatic balancing procedures are completely inapplicable to this power plant.

### 2. PARTS & MANUAL DATA
* **Status:** Incompatible configuration request."""
            response_placeholder.write(assistant_response)
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            st.stop()

        # SCENARIO D: Execution Loop
        else:
            with st.spinner("Executing mathematical spatial context scan..."):
                try:
                    context_str = "No directly matching documentation found in database."
                    
                    # INJECT MEMORY INTO DATABASE LOOKUP
                    search_query = user_query
                    if st.session_state.active_topic:
                        search_query = f"{st.session_state.active_topic} {user_query}"
                    
                    if st.session_state.vector_index is not None and len(st.session_state.vector_metadata) > 0:
                        query_vector = np.array([get_embedding(search_query)]).astype('float32')
                        distances, indices = st.session_state.vector_index.search(query_vector, 12)
                        
                        matched_chunks = []
                        for score, idx in zip(distances[0], indices[0]):
                            if idx != -1 and idx < len(st.session_state.vector_metadata):
                                if score < 1.3:
                                    chunk_data = st.session_state.vector_metadata[idx]
                                    matched_chunks.append(f"Source: {chunk_data['source']} - Page {chunk_data['page']}\nContent: {chunk_data['text']}")
                        
                        if matched_chunks:
                            context_str = "\n\n---\n\n".join(matched_chunks)
                    else:
                        context_str = f"System structural configuration info: No technical manual documentation PDFs have been vectorized or uploaded into server memory yet via the administrative workbench panel interface."

                    # Update internal tracking state with estimated query overhead
                    st.session_state.daily_token_consumption += 9000

                    # Adjust prompt rules to explicitly reinforce persistent topic memory
                    topic_context_injection = ""
                    if st.session_state.active_topic:
                        topic_context_injection = f"The technician is explicitly working within the sub-context domain of: {st.session_state.active_topic}. Keep all tool lists, part numbers, and steps locked strictly onto this operational procedure."

                    final_prompt = f"""You are supporting a licensed aircraft maintenance technician.
You must answer the user's question relying EXCLUSIVELY on the provided manual extracts below.

CRITICAL WORKSPACE LIMITATION:
You are explicitly assigned to find information ONLY for the following engine profile baseline: ROTAX {st.session_state.active_engine}.
If the provided manual extracts contain text, tables, or notes that explicitly state they belong to a DIFFERENT engine type (e.g., if your active profile is 916IS, but the text says "for 912/914 Series only" or references "carburetors/floats"), you must STRICTLY IGNORE and completely discard that data. Do not list those parts, specs, or troubleshooting notes anywhere in your response.

CRITICAL DISCIPLINE DIRECTIVE FOR TECHNICAL SUPPORT:
1. Your primary purpose is to help the user complete maintenance tasks SAFELY and SUCCESSFULLY right now. 
2. NEVER copy or output generic sentences that tell the user to \"refer to the maintenance manual\" or \"see Chapter X\". You are their interface to the manual. You must extract and output the actual, physical, sequential step-by-step instructions contained in the text.
3. If the provided manual extracts contain the actual steps, tolerances, clearances, or values for your active engine model, you MUST write them out in explicit detail under Section 1 so the technician can complete the activity without opening another file.
4. STRICT FILTERING FOR SECTION 2: Do NOT dump generic parts lists, internal sub-components, or exploded-view catalogs (e.g., float pins, internal gaskets, tiny washers) unless the manual explicitly states they must be *replaced* or *installed* during this specific maintenance task. Focus exclusively on the external tools, gauges, consumables (like specified Loctite grades or oils), and major line kits required to perform the workflow.
5. STRING OUTPUT RESTRICTION: The manual extracts may contain the literal phrase \"Clarification required from user\" or automated checklist questions embedded in troubleshooting tables. Do NOT copy or output this phrase word-for-word into Section 1 or Section 2 unless you are genuinely unable to answer the user's question based on the text. Filter it out of your raw text extractions.

Structure your response exactly like this:

### 1. QUICK SPEC / PROCEDURE
* Provide the concrete, sequential maintenance steps, checks, settings, or technical values extracted from the text below. Write them out fully so the technician can perform the work safely.
* If the task text is missing from the extracts, explicitly ask a clear technical clarifying question to narrow down the missing details.

### 2. PARTS & MANUAL DATA
* List specific part numbers, tool codes, or official manual chapter titles explicitly extracted from the text. Filter strictly according to Directive 4.
* If missing due to text gaps, state: \"Manual data gaps present\".

---
MANUAL EXTRACTS:
{context_str}
---
USER QUESTION: {user_query}"""

                    assistant_response = call_llm(final_prompt)
                    response_placeholder.write(assistant_response)
                    
                except Exception as e:
                    assistant_response = f"An error occurred during matrix processing: {str(e)}"
                    response_placeholder.error(assistant_response)
                    
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})