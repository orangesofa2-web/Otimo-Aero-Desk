import streamlit as st
from pypdf import PdfReader
import os
import re
import requests
import json
import numpy as np
from openai import OpenAI
import faiss

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Otimo Aero AI Desk",
    page_icon="✈️",
    layout="wide"
)

# =====================================================
# 2. DUAL API CONFIGURATION SAFETY GATES
# =====================================================
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")

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
# 3. SESSION STATE INITIALIZATION
# =====================================================
if "documents" not in st.session_state:
    st.session_state.documents = []

# Load local vector index into live container memory on startup if it exists
if "vector_index" not in st.session_state:
    if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
        try:
            st.session_state.vector_index = faiss.read_index(INDEX_PATH)
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                st.session_state.vector_metadata = json.load(f)
            # Reconstruct indexed documents list from metadata trace
            st.session_state.documents = list(set(m["source"] for m in st.session_state.vector_metadata))
        except Exception:
            st.session_state.vector_index = None
            st.session_state.vector_metadata = []
    else:
        st.session_state.vector_index = None
        st.session_state.vector_metadata = []

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello. True semantic vector production engine active. Enter your technical query below for high-fidelity maintenance support."
        }
    ]

if "pending_clarification" not in st.session_state:
    st.session_state.pending_clarification = None

# =====================================================
# 4. TECHNICAL SAFETY LAYERS & CORE ENGINES
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

# Helper: Request Dense Vector Coordinates from OpenAI Embeddings Engine
def get_embedding(text: str, model="text-embedding-3-small"):
    cleaned_text = text.replace("\n", " ")
    response = openai_client.embeddings.create(input=[cleaned_text], model=model)
    return response.data[0].embedding

# =====================================================
# 5. DOCUMENT INGESTION & VECTOR MATRIX BUILDER
# =====================================================
def rebuild_vector_database(uploaded_files):
    all_chunks = []
    
    # Process files locally page-by-page to keep contextual bounds intact
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
            # Build the 1536-dimensional FAISS Euclidean Distance Matrix
            dimension = len(embeddings_list[0])
            np_embeddings = np.array(embeddings_list).astype('float32')
            
            index = faiss.IndexFlatL2(dimension)
            index.add(np_embeddings)
            
            # Save the files directly to the local Streamlit repository instance
            faiss.write_index(index, INDEX_PATH)
            with open(METADATA_PATH, "w", encoding="utf-8") as f:
                json.dump(metadata_list, f, ensure_ascii=False, indent=2)
                
            st.session_state.vector_index = index
            st.session_state.vector_metadata = metadata_list
            st.session_state.documents = list(set(m["source"] for m in metadata_list))
            st.success("Universal semantic database built and stored successfully!")
            st.rerun()

# =====================================================
# 6. OPENROUTER PRODUCTION HANDSHAKE (LLAMA 3.1 8B)
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
# 7. SIDEBAR CONTROL PANEL
# =====================================================
with st.sidebar:
    st.header("Manual Management")
    uploaded_files = st.file_uploader(
        "Upload Technical Manuals",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        current_uploads = set(f.name for f in uploaded_files)
        existing_sources = set(st.session_state.documents)
        
        # Trigger vector calculation only if the file upload state changes
        if current_uploads != existing_sources:
            with st.spinner("Executing high-dimensional conceptual indexing..."):
                rebuild_vector_database(uploaded_files)
                
    # Reset local cache index files if file tray is cleared completely
    if not uploaded_files and st.session_state.vector_metadata:
        if os.path.exists(INDEX_PATH): os.remove(INDEX_PATH)
        if os.path.exists(METADATA_PATH): os.remove(METADATA_PATH)
        st.session_state.vector_index = None
        st.session_state.vector_metadata = []
        st.session_state.documents = []
        st.rerun()

    st.divider()
    st.metric("Indexed Manuals", len(st.session_state.documents))
    st.metric("Searchable Vector Coordinates", len(st.session_state.vector_metadata) if st.session_state.vector_metadata else 0)

# =====================================================
# 8. MAIN CHAT DISPLAY
# =====================================================
st.title("Otimo Aero")
st.subheader("Next-Generation Aviation Technical AI Desk")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# =====================================================
# 9. USER COMMAND RUNNER
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
            with st.spinner("Executing mathematical spatial context scan..."):
                try:
                    context_str = "No directly matching documentation found in database."
                    
                    if st.session_state.vector_index is not None and len(st.session_state.vector_metadata) > 0:
                        # Vectorise the user's incoming query text using OpenAI Embeddings API
                        query_vector = np.array([get_embedding(user_query)]).astype('float32')
                        
                        # Query the local FAISS matrix for the 12 closest matching page blocks conceptually
                        distances, indices = st.session_state.vector_index.search(query_vector, 12)
                        
                        matched_chunks = []
                        for score, idx in zip(distances[0], indices[0]):
                            if idx != -1 and idx < len(st.session_state.vector_metadata):
                                # Ambiguity Gate: If distance score is too wide, discard as background noise
                                if score < 1.3:
                                    chunk_data = st.session_state.vector_metadata[idx]
                                    matched_chunks.append(f"Source: {chunk_data['source']} (Page {chunk_data['page']})\nContent: {chunk_data['text']}")
                        
                        if matched_chunks:
                            context_str = "\n\n---\n\n".join(matched_chunks)

                    # Ironclad Prompt Structure
                    final_prompt = f"""You are supporting a licensed aircraft maintenance technician.
You must answer the user's question relying EXCLUSIVELY on the provided manual extracts below.

CRITICAL DISCIPLINE DIRECTIVE FOR TECHNICAL SUPPORT:
1. Your primary purpose is to help the user complete maintenance tasks SAFELY and SUCCESSFULLY right now. 
2. NEVER copy or output generic sentences that tell the user to \"refer to the maintenance manual\" or \"see Chapter X\". You are their interface to the manual. You must extract and output the actual, physical, sequential step-by-step instructions contained in the text.
3. If the provided manual extracts contain the actual steps, tolerances, clearances, or values, you MUST write them out in explicit detail under Section 1 so the technician can complete the activity without opening another file.
4. IF AND ONLY IF the explicit step-by-step physical procedure or target values are entirely absent or cut off within the extracts below, you must NOT invent data or give vague summaries. Instead, use Section 1 to ask a simple, precise clarifying question to get the missing context or component name needed to pull the correct pages.

Structure your response exactly like this:

### 1. QUICK SPEC / PROCEDURE
* Provide the concrete, sequential maintenance steps, checks, settings, or technical values extracted from the text below. Write them out fully so the technician can perform the work safely.
* If the task text is missing from the extracts, explicitly ask a clear technical clarifying question to narrow down the missing details.

### 2. PARTS & MANUAL DATA
* List specific part numbers, tool codes, or official manual chapter titles explicitly extracted from the text.
* If missing due to text gaps, state: \"Clarification required from user\".

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