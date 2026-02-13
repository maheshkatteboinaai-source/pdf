import streamlit as st
import os
import asyncio

from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------------------------------------------
# FIX: Create event loop for Python 3.13 + Streamlit
# ---------------------------------------------------
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# ---------------------------------------------------
# GET GOOGLE API KEY
# ---------------------------------------------------
def get_api_key():
    return (
        st.secrets.get("GOOGLE_API_KEY")
        if "GOOGLE_API_KEY" in st.secrets
        else os.getenv("GOOGLE_API_KEY")
    )

# ---------------------------------------------------
# LOAD GEMINI LLM
# ---------------------------------------------------
def get_llm():
    api_key = get_api_key()

    if not api_key:
        st.error("Google API key not found")
        st.stop()

    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.7
    )

# ---------------------------------------------------
# GENERATE RESPONSE
# ---------------------------------------------------
def generate_answer(prompt):
    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        return response.content

    except Exception as e:
        return f"Error: {str(e)}"

# ---------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------
st.title("📄 SlideSense AI Assistant")

prompt = st.text_area("Enter your question")

if st.button("Generate"):
    if prompt.strip():
        with st.spinner("Thinking..."):
            result = generate_answer(prompt)

        st.subheader("Answer")
        st.write(result)
    else:
        st.warning("Please enter a prompt")
