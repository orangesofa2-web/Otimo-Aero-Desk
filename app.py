import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
import os
import re
from collections import Counter

# 1. Page Configuration
st.set_page_config(
    page_title="Otimo Aero Technical Desk",
    page_icon="✈️",
    layout="wide"
)

# 2. Configure Gemini API
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
elif os.environ.get("GEMINI_API_KEY"):
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
else:
    st.error("Missing Gemini API Key. Please add it to your Streamlit Secrets.")
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

def split_into_chunks(text, size=500):
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
st.subheader("Technical Support Desk v2")

# 5. Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "Hello. Local semantic engine ready. Drop maintenance books or specs in the sidebar for concise, direct answers."
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
        with st.spinner("Extracting matching data segments..."):
            try:
                model = genai.GenerativeModel("gemini-2.5-flash")
                
                query_profile = get_text_profile(user_query)
                scored_chunks = []
                
                for item in st.session_state.document_registry:
                    score = score_chunk_universally(item["profile"], query_profile)
                    if score > 0:
                        scored_chunks.append((score, item["text"]))
                
                scored_chunks.sort(key=lambda x: x[0], reverse=True)
                top_context = [chunk for score, chunk in scored_chunks[:5]]
                context_str = "\n---\n".join(top_context)
                
                # Sharp, ultra-concise prompt structure using direct bullet points
                full_prompt = f"""
                You are the expert AI technical assistant for Otimo Aero. 
                You must be extremely concise, direct, and practical. No conversational filler or fluff.
                
                Structure your answer exactly like this:
                
                ### 1. QUICK SPEC / PROCEDURE
                * Give the direct answer, tool, or physical process immediately using bullet points.
                * Keep safety parameters or torque limits to 1-2 sharp lines.
                
                ### 2. PARTS & MANUAL DATA
                * Extract only the exact part numbers, consumables, or manual chapters found in the text below. 
                * If the specific part/paste name isn't mentioned in the text, state: "Not in uploaded files (using baseline)."
                
                ---
                MANUAL EXTRACTS:
                {context_str if context_str else 'No direct documentation matches.'}
                ---
                
                USER QUESTION: {user_query}
                """
                
                response = model.generate_content(full_prompt, generation_config={"temperature": 0.2})
                assistant_response = response.text
                response_placeholder.write(assistant_response)
                
            except Exception as e:
                assistant_response = f"An error occurred: {str(e)}"
                response_placeholder.error(assistant_response)
                
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})