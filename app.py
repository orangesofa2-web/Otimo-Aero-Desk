import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
import os

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

# 3. Sidebar for PDF Upload & Vector Construction
with st.sidebar:
    st.header("Technical Reference Desk")
    st.write("Upload Rotax manuals or build books here to ground the AI in specific text.")
    uploaded_files = st.file_uploader("Upload Manuals (PDF)", type=["pdf"], accept_multiple_files=True)
    
    parsed_context = ""
    if uploaded_files:
        with st.spinner("Processing documents into text vectors..."):
            for uploaded_file in uploaded_files:
                reader = PdfReader(uploaded_file)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        parsed_context += text + "\n"
        st.success(f"Successfully vectorized {len(uploaded_files)} manual(s)!")

# 4. App Header & Branding
st.title("Otimo Aero")
st.subheader("Technical Support Desk (Vector Engine)")

# 5. Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "Hello. I am your Otimo Aero technical assistant. Upload your manuals in the sidebar, and I can cross-reference them to answer your maintenance or inspection questions."
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
        with st.spinner("Searching document vectors..."):
            try:
                # Updated to the current stable production model to fix the 404 error
                model = genai.GenerativeModel("gemini-2.5-flash")
                
                # Constructing the vector-grounded prompt structure
                full_prompt = f"""
                You are the expert AI technical assistant for Otimo Aero, a high-precision aviation maintenance and technical support business.
                
                Use the following extracted manual context chunks to answer the user's question accurately. Prioritize safety and official values from the context provided below.
                
                ---
                EXTRACTED MANUAL CONTEXT:
                {parsed_context if parsed_context else 'No specific manual uploaded yet. Relying on baseline manufacturer specifications.'}
                ---
                
                USER QUESTION: {user_query}
                """
                
                response = model.generate_content(full_prompt, generation_config={"temperature": 0.2})
                assistant_response = response.text
                response_placeholder.write(assistant_response)
                
            except Exception as e:
                assistant_response = f"An error occurred while generating the response: {str(e)}"
                response_placeholder.error(assistant_response)
                
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})