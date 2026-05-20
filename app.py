import streamlit as st
from pypdf import PdfReader
import os
import re
import math
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

# Helper: Tokenize text into clean alphanumeric lowercase words
def tokenize(text):
    return re.findall(r'\b[a-z0-9]{3,20}\b', text.lower())

# 3. Dynamic Access Control (Checks URL for ?admin=true)
is_admin = st.query_params.get("admin") == "true"

if is_admin:
    with st.sidebar:
        st.header("Admin Control: Reference Desk")
        st.write("Upload manuals here. This sidebar is hidden from regular web traffic.")
        uploaded_files = st.file_uploader("Upload Manuals (PDF)", type=["pdf"], accept_multiple_files=True)
        
        if "document_registry" not in st.session_state:
            st.session_state.document_registry = []
        if "uploaded_filenames" not in st.session_state:
            st.session_state.uploaded_filenames = []
        if "global_word_counts" not in st.session_state:
            st.session_state.global_word_counts = Counter()

        if not uploaded_files and st.session_state.uploaded_filenames:
            st.session_state.document_registry = []
            st.session_state.uploaded_filenames = []
            st.session_state.global_word_counts = Counter()

        if uploaded_files:
            current_names = [f.name for f in uploaded_files]
            if any(name not in current_names for name in st.session_state.uploaded_filenames):
                st.session_state.document_registry = []
                st.session_state.uploaded_filenames = []
                st.session_state.global_word_counts = Counter()
                
            new_files = [f for f in uploaded_files if f.name not in st.session_state.uploaded_filenames]
            if new_files:
                with st.spinner("Building local dynamic indexing matrix..."):
                    for uploaded_file in new_files:
                        try:
                            reader = PdfReader(uploaded_file)
                            for page_num, page in enumerate(reader.pages):
                                text = page.extract_text()
                                if text:
                                    tokens = tokenize(text)
                                    if tokens:
                                        page_word_counts = Counter(tokens)
                                        for word in page_word_counts.keys():
                                            st.session_state.global_word_counts[word] += 1
                                            
                                        st.session_state.document_registry.append({
                                            "text": text,
                                            "word_counts": page_word_counts,
                                            "total_words": len(tokens),
                                            "source": f"{uploaded_file.name} (Page {page_num + 1})"
                                        })
                            st.session_state.uploaded_filenames.append(uploaded_file.name)
                        except Exception as parse_err:
                            st.error(f"Error parsing {uploaded_file.name}: {str(parse_err)}")
                st.success(f"Successfully indexed {len(st.session_state.uploaded_filenames)} manuals!")
else:
    if "document_registry" not in st.session_state:
        st.session_state.document_registry = []
    if "global_word_counts" not in st.session_state:
        st.session_state.global_word_counts = Counter()

# 4. App Header & Branding
st.title("Otimo Aero")
st.subheader("Technical Support Desk (OpenRouter Production Engine)")

# 5. Initialize Chat History & Context Memory State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "Hello. Production engine active. Enter your technical query below for unthrottled, precise maintenance support."
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
        
        # SCENARIO A: Resolving an active clarification request
        if st.session_state.pending_clarification:
            original_intent = st.session_state.pending_clarification
            st.session_state.pending_clarification = None  # Reset flag
            
            user_query = f"{original_intent} for {user_query}"
            clean_q = user_query.lower().replace(" ", "").replace("-", "")
            
        # SCENARIO B: Catching a vague engine term that requires a qualifying question
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
            with st.spinner("Scanning manual indices..."):
                try:
                    query_tokens = tokenize(user_query)
                    total_pages_in_registry = len(st.session_state.document_registry)
                    
                    matched_pages = []
                    
                    if total_pages_in_registry > 0 and query_tokens:
                        for item in st.session_state.document_registry:
                            page_score = 0.0
                            page_word_counts = item["word_counts"]
                            total_words = item["total_words"]
                            
                            for token in query_tokens:
                                if token in page_word_counts:
                                    tf = page_word_counts[token] / total_words
                                    docs_with_token = st.session_state.global_word_counts.get(token, 1)
                                    idf = math.log(total_pages_in_registry / docs_with_token) + 1.0
                                    page_score += (tf * idf)
                            
                            if page_score > 0:
                                matched_pages.append((page_score, item["text"], item["source"]))
                    
                    # Sort pages by mathematical TF-IDF score
                    matched_pages.sort(key=lambda x: x[0], reverse=True)
                    
                    # AMBIGUITY GATE CHECK:
                    # If the user asks a question but the keywords don't match any page with confidence,
                    # or if the query is fundamentally too broad, trigger immediate clarification request.
                    if not matched_pages or (len(matched_pages) > 1 and matched_pages[0][0] < 0.01):
                        context_str = "No directly matching documentation found or query context is highly ambiguous."
                    else:
                        top_context = [text for score, text, source in matched_pages[:8]]
                        context_str = "\n---\n".join(top_context)
                    
                    full_prompt = f"""You are the technical AI desk assistant for Otimo Aero, indexing official technical aircraft documentation.
You output answers in a strict, professional, itemized layout. No conversational fluff, assumptions, or external baseline guesses.

CRITICAL DISCIPLINE DIRECTIVE FOR AIRWORTHINESS SAFETY:
* You must answer the user's question relying EXCLUSIVELY on the provided manual extracts below.
* IF THE USER'S PROMPT IS AMBIGUOUS, OR IF THE PROVIDED EXTRACTS DO NOT CONTAIN AN EXACT, DEFINITIVE, UNAMBIGUOUS PROCEDURE MATCH FOR THE SPECIFIC SYSTEM ENQUIRED ABOUT, YOU MUST STOP.
* If there is any ambiguity, you must NOT provide generic steps. Instead, use section 1 to ask a highly specific technical clarifying question to narrow down the precise component reference, chapter title, or parameter needed.

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