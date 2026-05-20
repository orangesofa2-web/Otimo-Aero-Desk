import streamlit as st
from pypdf import PdfReader
import os
import re
import json
import requests
import numpy as np
from openai import OpenAI
import faiss

# 1. Page Configuration
st.set_page_config(
    page_title="Otimo Aero Technical Desk",
    page_icon="✈️",
    layout="wide"
)

# 2. Configure Dual API Keys Safety Gates
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")

if not OPENROUTER_API_KEY or not OPENAI_API_KEY:
    st.error("Missing required API credentials in Streamlit Secrets. Ensure both OPENROUTER_API_KEY and OPENAI_API_KEY are configured.")
    st.stop()

# Initialize OpenAI Client for Universal Semantic Embeddings
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Index File Paths for Persistent Storage
INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "faiss_metadata.json"

# Helper: Semantic Text Chunking Engine
def chunk_text(text, source_name, chunk_size=800, overlap=150):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        chunks.append({
            "text": chunk_text,
            "source": source_name
        })
        i += (chunk_size - overlap)
    return chunks

# Helper: Fetch Dense Vector Coordinates from OpenAI
def get_embedding(text, model="text-embedding-3-small"):
    # Clean up newline breaks to stabilize vector output
    cleaned_text = text.replace("\n", " ")
    response = openai_client.embeddings.create(input=[cleaned_text], model=model)
    return response.data[0].embedding

# 3. Dynamic Access Control & Ingestion Core (URL ?admin=true)
is_admin = st.query_params.get("admin") == "true"

# Load Index & Metadata into Memory on Startup if they exist
if "vector_index" not in st.session_state:
    if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
        try:
            st.session_state.vector_index = faiss.read_index(INDEX_PATH)
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                st.session_state.vector_metadata = json.load(f)
        except Exception:
            st.session_state.vector_index = None
            st.session_state.vector_metadata = []
    else:
        st.session_state.vector_index = None
        st.session_state.vector_metadata = []

if is_admin:
    with st.sidebar:
        st.header("Admin Control: Vector Desk")
        st.write("Upload or refresh your technical library manuals here.")
        uploaded_files = st.file_uploader("Upload Manuals (PDF)", type=["pdf"], accept_multiple_files=True)
        
        # Clear database if files are removed entirely from the tray
        if not uploaded_files and st.session_state.vector_metadata:
            if os.path.exists(INDEX_PATH): os.remove(INDEX_PATH)
            if os.path.exists(METADATA_PATH): os.remove(METADATA_PATH)
            st.session_state.vector_index = None
            st.session_state.vector_metadata = []
            st.rerun()

        if uploaded_files:
            # Check if we need to process new additions
            existing_sources = set(m["source"].split(" (Page")[0] for m in st.session_state.vector_metadata)
            current_uploads = set(f.name for f in uploaded_files)
            
            if current_uploads != existing_sources:
                with st.spinner("Processing deep vectorization across library..."):
                    all_chunks = []
                    for uploaded_file in uploaded_files:
                        try:
                            reader = PdfReader(uploaded_file)
                            for page_num, page in enumerate(reader.pages):
                                page_text = page.extract_text()
                                if page_text and len(page_text.strip()) > 50:
                                    file_label = f"{uploaded_file.name} (Page {page_num + 1})"
                                    all_chunks.extend(chunk_text(page_text, file_label))
                        except Exception as e:
                            st.error(f"Error parsing {uploaded_file.name}: {str(e)}")
                    
                    if all_chunks:
                        embeddings_list = []
                        metadata_list = []
                        progress_bar = st.progress(0)
                        
                        for idx, chunk in enumerate(all_chunks):
                            try:
                                vec = get_embedding(chunk["text"])
                                embeddings_list.append(vec)
                                metadata_list.append(chunk)
                            except Exception as api_err:
                                st.error(f"Embedding failure on chunk {idx}: {str(api_err)}")
                            progress_bar.progress((idx + 1) / len(all_chunks))
                        
                        if embeddings_list:
                            # Build the high-dimensional FAISS CPU matrix index (1536 dimensions)
                            dimension = len(embeddings_list[0])
                            np_embeddings = np.array(embeddings_list).astype('float32')
                            
                            index = faiss.IndexFlatL2(dimension)
                            index.add(np_embeddings)
                            
                            # Persist the calculations cleanly to the deployment disk container
                            faiss.write_index(index, INDEX_PATH)
                            with open(METADATA_PATH, "w", encoding="utf-8") as f:
                                json.dump(metadata_list, f, ensure_ascii=False, indent=2)
                                
                            st.session_state.vector_index = index
                            st.session_state.vector_metadata = metadata_list
                            st.success(f"Successfully vectorized {len(current_uploads)} manuals into permanent memory!")
                            st.rerun()

# 4. App Header & Branding
st.title("Otimo Aero")
st.subheader("Technical Support Desk (Dense Vector RAG Production Engine)")

# 5. Initialize Chat History & Context Memory State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "Hello. Vectorized production engine active. Enter your technical query below for high-fidelity maintenance support."
        }
    ]
if "pending_clarification" not in st.session_state:
    st.session_state.pending_clarification = None

# 6. Display Existing Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 7. Handle User Input and Generate Response
if user_query := st.chat_input("Enter your technical question here..."):
    
    with st.chat_message("user"):
        st.write(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        clean_q = user_query.lower().replace(" ", "").replace("-", "")
        
        # SCENARIO A: Handling an active clarification request
        if st.session_state.pending_clarification:
            original_intent = st.session_state.pending_clarification
            st.session_state.pending_clarification = None  # Reset flag
            user_query = f"{original_intent} specifically regarding {user_query}"
            clean_q = user_query.lower().replace(" ", "").replace("-", "")
            
        # SCENARIO B: Enforcing variant specification for broad engine lookups
        if "912" in clean_q and not any(v in clean_q for v in ["uls", "ul", "is"]):
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

        # HARDCODED FUEL-INJECTION SECURITY GUARDS
        is_carb_query = any(x in clean_q for x in ["carb", "sync", "balance", "float", "choke"])
        is_injected_engine = any(x in clean_q for x in ["915", "916", "912is"])
        
        if is_carb_query and is_injected_engine:
            assistant_response = """### 1. QUICK SPEC / PROCEDURE
* **CRITICAL ERROR:** The engine model specified (Rotax fuel-injected iS series) utilizes dual-channel electronic fuel injection and does not possess carburetors.
* Carburetor synchronization and pneumatic balancing procedures are completely inapplicable to this power plant.

### 2. PARTS & MANUAL DATA
* **Status:** Incompatible configuration request."""
            response_placeholder.write(assistant_response)
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        
        # SCENARIO C: Standard Vector-Search Path
        else:
            with st.spinner("Executing mathematical spatial context scan..."):
                try:
                    context_str = "No directly matching documentation found in database."
                    
                    if st.session_state.vector_index is not None and len(st.session_state.vector_metadata) > 0:
                        # Vectorize the user's incoming query string
                        query_vector = np.array([get_embedding(user_query)]).astype('float32')
                        
                        # Query the FAISS index for the 6 closest matching chunks mathematically
                        distances, indices = st.session_state.vector_index.search(query_vector, 6)
                        
                        matched_chunks = []
                        for score, idx in zip(distances[0], indices[0]):
                            # -1 indicates no match found in FAISS pool bounds
                            if idx verifiable_index != -1 and idx < len(st.session_state.vector_metadata):
                                # ENFORCE AMBIGUITY GATE: If geometric similarity distance is too wide,
                                # treat chunk as background noise to prevent hallucinations.
                                if score < 1.2: 
                                    chunk_data = st.session_state.vector_metadata[idx]
                                    matched_chunks.append(f"Source: {chunk_data['source']}\nContent: {chunk_data['text']}")
                        
                        if matched_chunks:
                            context_str = "\n\n---\n\n".join(matched_chunks)
                    
                    # Direct System prompt configuration with ironclad guardrails
                    full_prompt = f"""You are the technical AI desk assistant for Otimo Aero, indexing official technical aircraft documentation.
You output answers in a strict, professional, itemized layout. No conversational fluff, assumptions, or external baseline guesses.

CRITICAL DISCIPLINE DIRECTIVE FOR AIRWORTHINESS SAFETY:
* You must answer the user's question relying EXCLUSIVELY on the provided manual extracts below.
* IF THE USER'S PROMPT IS AMBIGUOUS, OR IF THE PROVIDED EXTRACTS DO NOT CONTAIN AN EXACT, DEFINITIVE, UNAMBIGUOUS PROCEDURE MATCH FOR THE SPECIFIC SYSTEM ENQUIRED ABOUT, YOU MUST STOP.
* If there is any ambiguity, you must NOT provide generic steps. Instead, use section 1 to ask a highly specific technical clarifying question to narrow down the precise component reference, chapter title, or parameter needed.
* DO NOT ask the user for an engine serial number unless it is explicitly requested by the text for safety tolerances. The model variant provided by the user is sufficient.

Structure your response exactly like this:

### 1. QUICK SPEC / PROCEDURE
* Provide the direct maintenance steps or technical values extracted from the text below. 
* IF AMBIGUOUS OR DATA IS INSUFFICIENT, explicitly ask the user for the specific missing context or component identification needed to guarantee an accurate match.

### 2. PARTS & MANUAL DATA
* List specific part numbers, tool codes, or manual chapter titles extracted from the text.
* If missing due to an ambiguous query or text gaps, state: "Clarification required from user".

---
MANUAL EXTRACTS:
{context_str}
---
USER QUESTION: {user_query}"""

                    url = "https://openrouter.ai/api/v1/chat/completions"
                    headers = {
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json"
                    }
                    
                    data = {
                        "model": "meta-llama/llama-3.1-8b-instruct",
                        "messages": [{"role": "user", "content": full_prompt}],
                        "temperature": 0.0,
                        "providers": {
                            "order": ["Lepton", "Together"],
                            "allow_fallbacks": True
                        }
                    }
                    
                    res = requests.post(url, json=data, headers=headers)
                    if res.status_code == 200:
                        assistant_response = res.json()["choices"][0]["message"]["content"]
                        response_placeholder.write(assistant_response)
                    else:
                        assistant_response = f"OpenRouter Connection Error ({res.status_code}): {res.text}"
                        response_placeholder.error(assistant_response)
                        
                except Exception as e:
                    assistant_response = f"An error occurred: {str(e)}"
                    response_placeholder.error(assistant_response)
                    
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})