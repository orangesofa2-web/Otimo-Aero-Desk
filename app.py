import streamlit as st
from pypdf import PdfReader
import os
import re
import requests

# 1. Page Configuration
st.set_page_config(
    page_title="Otimo Aero Technical Desk",
    page_icon="✈️",
    layout="wide"
)

# 2. Configure Single OpenRouter API Key Safety Gate
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    st.error("Missing OPENROUTER_API_KEY in Streamlit Secrets. Please add it to your app settings.")
    st.stop()

# Helper: Tokenize text into clean lowercase alphanumeric tokens
def tokenize(text):
    return re.findall(r'\b[a-z0-9]{3,20}\b', text.lower())

# 3. Dynamic Access Control & Ingestion Core (URL ?admin=true)
is_admin = st.query_params.get("admin") == "true"

if "document_registry" not in st.session_state:
    st.session_state.document_registry = []
if "uploaded_filenames" not in st.session_state:
    st.session_state.uploaded_filenames = []

if is_admin:
    with st.sidebar:
        st.header("Admin Control: Reference Desk")
        st.write("Upload or refresh your technical library manuals here.")
        uploaded_files = st.file_uploader("Upload Manuals (PDF)", type=["pdf"], accept_multiple_files=True)
        
        # Reset local cache if files are cleared from the tray
        if not uploaded_files and st.session_state.uploaded_filenames:
            st.session_state.document_registry = []
            st.session_state.uploaded_filenames = []
            st.rerun()

        if uploaded_files:
            current_names = [f.name for f in uploaded_files]
            if any(name not in current_names for name in st.session_state.uploaded_filenames):
                st.session_state.document_registry = []
                st.session_state.uploaded_filenames = []
                
            new_files = [f for f in uploaded_files if f.name not in st.session_state.uploaded_filenames]
            if new_files:
                with st.spinner("Indexing manual text streams..."):
                    for uploaded_file in new_files:
                        try:
                            reader = PdfReader(uploaded_file)
                            for page_num, page in enumerate(reader.pages):
                                page_text = page.extract_text()
                                if page_text and len(page_text.strip()) > 50:
                                    st.session_state.document_registry.append({
                                        "text": page_text,
                                        "source": f"{uploaded_file.name} (Page {page_num + 1})"
                                    })
                            st.session_state.uploaded_filenames.append(uploaded_file.name)
                        except Exception as e:
                            st.error(f"Error parsing {uploaded_file.name}: {str(e)}")
                    st.success(f"Successfully indexed {len(st.session_state.uploaded_filenames)} manuals!")
                    st.rerun()

# 4. App Header & Branding
st.title("Otimo Aero")
st.subheader("Technical Support Desk (Single-API Production Engine)")

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
        
        # SCENARIO C: Universal Multi-Pass Search
        else:
            with st.spinner("Analyzing manual layout filters..."):
                try:
                    # Filter out generic model keywords from the ranking math to stop context dilution
                    raw_tokens = tokenize(