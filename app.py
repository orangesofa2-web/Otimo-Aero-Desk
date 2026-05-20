import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
import os
import re

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
    st.error("Missing Gemini API Key.")
    st.stop()

# Helper function to split text into manageable sentences/paragraphs
def split_into_chunks(text, size=700):
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]

# Helper function to score chunks based on keyword matching (Local Indexing)
def find_relevant_chunks(chunks, query, max_results=5):
    query_words = re.findall(r'\w+', query.lower())
    scored_chunks = []
    for chunk in chunks:
        score = sum(1 for word in query_words if word in chunk.lower())
        if score > 0:
            scored_chunks.append((score, chunk))
    # Sort by highest match score
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return [chunk for score, chunk in scored_chunks[:max_results]]

# 3. Sidebar for PDF Upload & Token Reduction
with st.sidebar:
    st.header("Technical Reference Desk")
    st.write("Upload all technical manuals here. Processing is optimized to stay 100% free.")
    uploaded_files = st.file_uploader("Upload Manuals (PDF)", type=["pdf"], accept_multiple_files=True)
    
    if "document_chunks" not in st.session_state:
        st.session_state.document_chunks = []
        st.session_state.uploaded_filenames = []

    if uploaded_files:
        new_files = [f.name for f in uploaded_files if f.name not in st.session_state.uploaded_filenames]
        if new_files:
            with st.spinner("Analyzing and parsing text locally..."):
                for uploaded_file in uploaded_files:
                    if uploaded_file.name in st.session_state.uploaded_filenames:
                        continue
                    reader = PdfReader(uploaded_file)
                    file_text = ""
                    for page in reader.pages:
                        text = page.extract_text()
                        if text:
                            file_text += text + "\n"
                    
                    # Split into chunks locally
                    file_chunks = split_into_chunks(file_text)
                    st.session_state.document_chunks.extend(file_chunks)
                    st.session_state.uploaded_filenames.append(uploaded_file.name)
            st.success(f"Indexed {len(st.session_state.uploaded_filenames)} manual(s) successfully!")

# 4. App Header & Branding
st.title("Otimo Aero")
st.subheader("Technical Support Desk (Optimized Free-Tier Engine)")

# 5. Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "Hello. I am your Otimo Aero technical assistant. Upload all your manuals in the sidebar, and I will search them dynamically without hitting any limits."
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
                
                # Search the local index instead of dumping everything
                matched_segments = find_relevant_chunks(st.session_state.document_chunks, user_query)
                context_str = "\n---\n".join(matched_segments)
                
                full_prompt = f"""
                You are the expert AI technical assistant for Otimo Aero, a high-precision aviation maintenance and technical support business.
                
                Answer the user's question using the specific manual extracts provided below. If the extracts don't contain the answer, use your baseline manufacturer data to answer, but specify that it is baseline.
                
                EXTRACTED MANUAL CONTEXT:
                {context_str if context_str else 'No direct keyword matches found in uploaded manuals. Using baseline manufacturer guidelines.'}
                
                USER QUESTION: {user_query}
                """
                
                response = model.generate_content(full_prompt, generation_config={"temperature": 0.2})
                assistant_response = response.text
                response_placeholder.write(assistant_response)
                
            except Exception as e:
                assistant_response = f"An error occurred: {str(e)}"
                response_placeholder.error(assistant_response)
                
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})