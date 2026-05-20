import streamlit as st
from pypdf import PdfReader
import os
import re
from collections import Counter
import requests

# 1. Page Configuration
st.set_page_config(
    page_title="Otimo Aero Technical Desk",
    page_icon="✈️",
    layout="wide"
)

# 2. Configure OpenRouter API Key
OPENROUTER_API_KEY = ""
if "OPENROUTER_API_KEY" in st.secrets:
    OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
elif os.environ.get("OPENROUTER_API_KEY"):
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    st.error("Missing OPENROUTER_API_KEY in Streamlit Secrets. Please add it to your app settings.")
    st.stop()

# Helper: Clean text into character-grams to extract root meanings across technical terms
def get_text_profile(text):
    words = re.findall(r'\w+', text.lower())
    profile = Counter(words)
    for word in words:
        if len(word) > 3:
            for i in range(len(word) - 3):
                profile[word[i:i+4]] += 0.5
    return profile

# Helper: Universal local relevance scoring using vector-space token frequency overlap
def score_chunk_universally(chunk_profile, query_profile):
    intersection = set(chunk_profile.keys()) & set(query_profile.keys())
    score = sum(chunk_profile[token] * query_profile[token] for token in intersection)
    return score

def split_into_chunks(text, size=1000):
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]

# 3. Sidebar for PDF Upload & Index Processing
with st.sidebar:
    st.header("Technical Reference Desk")
    st.write("Upload manuals here. The local semantic engine scales automatically.")
    uploaded_files = st.file_uploader("Upload Manuals (PDF)", type=["pdf"], accept_multiple_files=True)
    
    if "document_registry" not in st.session_state:
        st.session_state.document_registry = []
    if "uploaded_filenames" not in st.session_state:
        st.session_state.uploaded_filenames = []

    if not uploaded_files and st.session_state.uploaded_filenames:
        st.session_state.document_registry = []
        st.session_state.uploaded_filenames = []

    if uploaded_files:
        current_names = [f.name for f in uploaded_files]
        if any(name not in current_names for name in st.session_state.uploaded_filenames):
            st.session_state.document_registry = []
            st.session_state.uploaded_filenames = []
            
        new_files = [f for f in uploaded_files if f.name not in st.session_state.uploaded_filenames]
        if new_files:
            with st.spinner("Building local semantic indices..."):
                for uploaded_file in new_files:
                    try:
                        reader = PdfReader(uploaded_file)
                        file_text = ""
                        for page in reader.pages:
                            text = page.extract_text()
                            if text:
                                file_text += text + "\n"
                        
                        file_chunks = split_into_chunks(file_text)
                        for chunk in file_chunks:
                            profile = get_text_profile(chunk)
                            st.session_state.document_registry.append({
                                "text": chunk,
                                "profile": profile
                            })
                        st.session_state.uploaded_filenames.append(uploaded_file.name)
                    except Exception as parse_err:
                        st.error(f"Error parsing {uploaded_file.name}: {str(parse_err)}")
            st.success(f"Indexed {len(st.session_state.uploaded_filenames)} files!")

# 4. App Header & Branding
st.title("Otimo Aero")
st.subheader("Technical Support Desk (OpenRouter Production Engine)")

# 5. Initialize Chat History & Context Memory State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "Hello. Production engine active. Drop your manuals in the sidebar for unthrottled, precise maintenance support."
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
        
        # Normalize text to catch variations cleanly
        clean_q = user_query.lower().replace(" ", "").replace("-", "")
        
        # SCENARIO A: Resolving an active clarification request
        if st.session_state.pending_clarification:
            original_intent = st.session_state.pending_clarification
            st.session_state.pending_clarification = None  # Reset flag
            
            # Reconstruct query using the chosen model variant
            user_query = f"{original_intent} for Rotax {user_query}"
            clean_q = user_query.lower().replace(" ", "").replace("-", "")
            
        # SCENARIO B: Catching a vague engine term that requires a qualifying question
        # Triggered if query references 912 broadly but skips the variant descriptor
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

        # HARDCODED MECHANICAL ENGINE GUARDS (Instant rejections)
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
        
        # SCENARIO C: All filters clear, execute standard manual search
        else:
            with st.spinner("Processing request via production gateway..."):
                try:
                    history_context = ""
                    if len(st.session_state.messages) > 2:
                        recent_messages = st.session_state.messages[-2:-1]
                        history_context = " ".join([m['content'] for m in recent_messages])
                    
                    combined_search_terms = f"{user_query} {history_context}"
                    query_profile = get_text_profile(combined_search_terms)
                    
                    scored_chunks = []
                    for item in st.session_state.document_registry:
                        score = score_chunk_universally(item["profile"], query_profile)
                        if score > 0:
                            scored_chunks.append((score, item["text"]))
                    
                    scored_chunks.sort(key=lambda x: x[0], reverse=True)
                    top_context = [chunk for score, chunk in scored_chunks[:10]]
                    context_str = "\n---\n".join(top_context)
                    
                    full_prompt = f"""You are the technical AI desk assistant for Otimo Aero, specializing in Rotax aircraft engines.
You output answers in a clean, professional, itemized layout. No conversational fluff or commentary.

REQUIRED CONSUMABLE BASELINE:
* If spark plug paste is queried but details are not found in the manuals below, explicitly output: Wacker Aerospace Heat Sink Paste P12 (Rotax P/N 897186). UK Price: £15.00 inc VAT. Do NOT suggest threadlockers.

Structure your response exactly like this:

### 1. QUICK SPEC / PROCEDURE
* Provide immediate, actionable maintenance steps or specifications based on the extracts below.
* Keep limits or safety figures to 1-2 sharp lines.

### 2. PARTS & MANUAL DATA
* List specific part numbers, part descriptions, or manual chapter references extracted from the text.

---
MANUAL EXTRACTS:
{context_str if context_str else 'No directly matching documentation found.'}
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
                        "temperature": 0.1
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