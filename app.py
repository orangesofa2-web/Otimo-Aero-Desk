import streamlit as st
from pypdf import PdfReader
import os
import re
import requests
from collections import Counter

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Otimo Aero AI Desk",
    page_icon="✈️",
    layout="wide"
)

# =====================================================
# 2. API CONFIGURATION
# =====================================================
OPENROUTER_API_KEY = (
    st.secrets.get("OPENROUTER_API_KEY")
    or os.environ.get("OPENROUTER_API_KEY")
)

if not OPENROUTER_API_KEY:
    st.error("Missing OPENROUTER_API_KEY in Streamlit Secrets. Please add it to your app settings.")
    st.stop()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# =====================================================
# 3. SESSION STATE INITIALIZATION
# =====================================================
if "documents" not in st.session_state:
    st.session_state.documents = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello. Production engine active. Enter your technical query below for high-fidelity maintenance support."
        }
    ]

if "pending_clarification" not in st.session_state:
    st.session_state.pending_clarification = None

# =====================================================
# 4. UTILITIES & TOKENIZERS
# =====================================================
def tokenize(text: str):
    return re.findall(r'\b[a-z0-9]{3,20}\b', text.lower())

# =====================================================
# 5. TECHNICAL SAFETY LAYER
# =====================================================
def requires_variant(query: str) -> bool:
    q = query.lower().replace(" ", "").replace("-", "")
    return "912" in q and not any(v in q for v in ["uls", "ul", "is"])

def invalid_configuration(query: str) -> bool:
    q = query.lower().replace(" ", "").replace("-", "")
    carb_terms = ["carb", "sync", "balance", "float", "choke"]
    injected_engines = ["915", "916", "912is"]
    
    is_carb_query = any(t in q for t in carb_terms)
    is_injected = any(e in q for e in injected_engines)
    return is_carb_query and is_injected

# =====================================================
# 6. RELEVANCE SEARCH FILTER ENGINE
# =====================================================
def retrieve_context(query: str, top_k: int = 12):
    raw_tokens = tokenize(query)
    noise_terms = {"912", "uls", "ul", "series", "type", "rotax", "engine"}
    important_tokens = [t for t in raw_tokens if t not in noise_terms]
    
    if not important_tokens:
        important_tokens = raw_tokens

    scored_chunks = []
    for chunk in st.session_state.chunks:
        text_lower = chunk["text"].lower()
        
        # High priority given to unique task actions (e.g., synchronization, torque)
        action_hits = sum(10.0 for token in important_tokens if token in text_lower)
        # Low baseline priority for generic model terms
        model_hits = sum(0.1 for token in raw_tokens if token in text_lower and token in noise_terms)
        
        total_score = action_hits + model_hits
        if total_score > 0:
            scored_chunks.append((total_score, chunk))

    # Sort strictly by technical task relevance
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return scored_chunks[:top_k]

# =====================================================
# 7. DOCUMENT INGESTION
# =====================================================
def ingest_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text and len(page_text.strip()) > 50:
            st.session_state.chunks.append({
                "text": page_text,
                "source": uploaded_file.name,
                "page": page_num + 1
            })

# =====================================================
# 8. PREMIUM OPENROUTER HANDSHAKE
# =====================================================
def call_llm(prompt: str):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "anthropic/claude-3.5-sonnet",
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
        ]
    }
    
    response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
    if response.status_code != 200:
        raise Exception(response.text)
    return response.json()["choices"][0]["message"]["content"]

# =====================================================
# 9. SIDEBAR CONTROL PANEL
# =====================================================
with st.sidebar:
    st.header("Manual Management")
    uploaded_files = st.file_uploader(
        "Upload Technical Manuals",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.documents:
                with st.spinner(f"Indexing {file.name}..."):
                    ingest_pdf(file)
                    st.session_state.documents.append(file.name)
        st.success("Manual indexing complete")

    st.divider()
    st.metric("Indexed Manuals", len(st.session_state.documents))
    st.metric("Searchable Page Units", len(st.session_state.chunks))

# =====================================================
# 10. MAIN CHAT DISPLAY
# =====================================================
st.title("Otimo Aero")
st.subheader("Next-Generation Aviation Technical AI Desk")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# =====================================================
# 11. USER COMMAND RUNNER
# =====================================================
user_query = st.chat_input("Enter technical maintenance question...")

if user_query:
    with st.chat_message("user"):
        st.write(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        clean_q = user_query.lower().replace(" ", "").replace("-", "")

        # SCENARIO A: Resolving pending clarification requests
        if st.session_state.pending_clarification:
            original_intent = st.session_state.pending_clarification
            st.session_state.pending_clarification = None
            user_query = f"{original_intent} specifically regarding {user_query}"
            clean_q = user_query.lower().replace(" ", "").replace("-", "")

        # SCENARIO B: Enforce specific engine variant selections
        if requires_variant(user_query):
            st.session_state.pending_clarification = user_query
            assistant_response = """### 🔍 SPECIFICATION REQUIRED
To provide the correct technical clearances or procedure parameters, please specify your exact engine model variant:
* **912 ULS** (100 hp, Carbureted)
* **912 UL** (80 hp, Carbureted)
* **912 iS** (100 hp, Fuel Injected)

*Please type your variant directly into the chat input below to proceed.*"""
            response_placeholder.write(assistant_response)
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            st.stop()

        # SCENARIO C: Hardcoded Configuration Blocking Guard
        if invalid_configuration(user_query):
            assistant_response = """### 1. QUICK SPEC / PROCEDURE
* **CRITICAL ERROR:** The engine model specified (Rotax fuel-injected iS series) utilizes dual-channel electronic fuel injection and does not possess carburetors.
* Carburetor synchronization and pneumatic balancing procedures are completely inapplicable to this power plant.

### 2. PARTS & MANUAL DATA
* **Status:** Incompatible configuration request."""
            response_placeholder.write(assistant_response)
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            st.stop()

        # SCENARIO D: Execution Loop
        else:
            with st.spinner("Retrieving technical procedures..."):
                try:
                    matches = retrieve_context(user_query)
                    
                    if not matches:
                        context_str = "No matching manual context found."
                    else:
                        context_blocks = []
                        for score, chunk in matches:
                            context_blocks.append(f"Source: {chunk['source']} (Page {chunk['page']})\nContent: {chunk['text']}")
                        context_str = "\n\n---\n\n".join(context_blocks)

                    # Ironclad Prompt Structure
                    final_prompt = f"""You are supporting a licensed aircraft maintenance technician.
You must answer the user's question relying EXCLUSIVELY on the provided manual extracts below.

CRITICAL DISCIPLINE DIRECTIVE FOR TECHNICAL SUPPORT:
1. Your primary purpose is to help the user complete maintenance tasks SAFELY and SUCCESSFULLY right now. 
2. NEVER copy or output generic sentences that tell the user to "refer to the maintenance manual" or "see Chapter X". You are their interface to the manual. You must extract and output the actual, physical, sequential step-by-step instructions contained in the text.
3. If the provided manual extracts contain the actual steps, tolerances, clearances, or values, you MUST write them out in explicit detail under Section 1 so the technician can complete the activity without opening another file.
4. IF AND ONLY IF the explicit step-by-step physical procedure or target values are entirely absent or cut off within the extracts below, you must NOT invent data or give vague summaries. Instead, use Section 1 to ask a simple, precise clarifying question to get the missing context or component name needed to pull the correct pages.

Structure your response exactly like this:

### 1. QUICK SPEC / PROCEDURE
* Provide the concrete, sequential maintenance steps, checks, settings, or technical values extracted from the text below. Write them out fully so the technician can perform the work safely.
* If the task text is missing from the extracts, explicitly ask a clear technical clarifying question to narrow down the missing details.

### 2. PARTS & MANUAL DATA
* List specific part numbers, tool codes, or official manual chapter titles explicitly extracted from the text.
* If missing due to text gaps, state: "Clarification required from user".

---
MANUAL EXTRACTS:
{context_str}
---
USER QUESTION: {user_query}"""

                    assistant_response = call_llm(final_prompt)
                    response_placeholder.write(assistant_response)
                    
                except Exception as e:
                    assistant_response = f"An error occurred: {str(e)}"
                    response_placeholder.error(assistant_response)
                    
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})