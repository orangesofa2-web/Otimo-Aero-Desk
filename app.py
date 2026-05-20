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

# 2. Configure OpenRouter API Key (Paid Unthrottled Production Tier)
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

# 3. Sidebar for PDF Upload & Index Processing with Deduplication
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

# 5. Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "Hello. Production engine active. Drop your manuals in the sidebar for unthrottled, precise maintenance support."
        }
    ]

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
        with st.spinner("Processing request via production gateway..."):
            try:
                # Maintain conversation continuity across follow-up queries locally
                history_context = ""
                if len(st.session_state.messages) > 2:
                    recent_messages = st.session_state.messages[-3:-1]
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
                
                # Production prompt architecture
                full_prompt = f"""
                You are the expert AI technical assistant for Otimo Aero. 
                You must be extremely concise, direct, and practical. No conversational filler or fluff.
                
                Contextual awareness: Resolve pronoun references (like "it", "this", "which paste") using the context history below.
                
                Structure your answer exactly like this:
                
                ### 1. QUICK SPEC / PROCEDURE
                * Give the direct answer, tool, or physical process immediately using bullet points.
                * Keep safety parameters or torque limits to 1-2 sharp lines.
                
                ### 2. PARTS & MANUAL DATA
                * Extract only the exact part numbers, consumables, or manual chapters found in the text below. 
                * If the specific part/paste name isn't explicitly mentioned in the text, state "Not in uploaded files" and immediately provide the industry/manufacturer baseline part number or spec anyway.
                
                ---
                RECENT RELEVANT CHAT HISTORY:
                {history_context if history_context else 'No prior history.'}
                ---
                MANUAL EXTRACTS:
                {context_str if context_str else 'No direct documentation matches.'}
                ---
                
                USER QUESTION: {user_query}
                """
                
                # Route data to OpenRouter endpoint using the robust Llama 3.1 8B model
                url = "https://openrouter.ai/api/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                }
                data = {
                    "model": "meta-llama/llama-3.1-8b-instruct",
                    "messages": [{"role": "user", "content": full_prompt}],
                    "temperature": 0.2
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