import streamlit as st
import google.generativeai as genai
import os

# 1. Page Configuration
st.set_page_config(
    page_title="Otimo Aero Technical Desk",
    page_icon="✈️",
    layout="centered"
)

# 2. Configure Gemini API
# Retrieves the key securely from Streamlit's Secrets management
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
elif os.environ.get("GEMINI_API_KEY"):
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
else:
    st.error("Missing Gemini API Key. Please add it to your Streamlit Secrets.")
    st.stop()

# 3. App Header & Branding
st.title("Otimo Aero")
st.subheader("Technical Support Desk")
st.write("Ask technical questions regarding Rotax maintenance, aircraft inspections, or build specifications.")

# 4. Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "Hello. I am your Otimo Aero technical assistant. How can I help you with your aircraft or engine maintenance tasks today?"
        }
    ]

# 5. Display Existing Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 6. Handle User Input and Generate Response
if user_query := st.chat_input("Enter your technical question here..."):
    
    # Display user's message
    with st.chat_message("user"):
        st.write(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Generate AI response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        with st.spinner("Consulting technical manuals..."):
            try:
                # Using the recommended model for general text tasks
                model = genai.GenerativeModel("models/gemini-1.5-flash")
                
                # Context to keep the AI grounded in your specific business domain
                system_instruction = (
                    "You are the expert AI technical assistant for Otimo Aero, a high-precision "
                    "aviation maintenance and technical support business specializing in Rotax engines "
                    "and light aircraft certification. Provide clear, professional, and technically accurate "
                    "guidance. Prioritize safety and adherence to official manuals."
                )
                
                # Formatting history for the API call
                history = [{"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]} for m in st.session_state.messages[:-1]]
                
                chat = model.start_chat(history=history)
                response = chat.send_message(user_query, generation_config={"temperature": 0.3})
                
                assistant_response = response.text
                response_placeholder.write(assistant_response)
                
            except Exception as e:
                assistant_response = f"An error occurred while generating the response: {str(e)}"
                response_placeholder.error(assistant_response)
                
    # Save assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})