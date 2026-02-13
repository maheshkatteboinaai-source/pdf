import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import asyncio
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
import time

# -------------------- Page Config --------------------
st.set_page_config(page_title="SlideSense", page_icon="📘", layout="wide")

# -------------------- Session --------------------
defaults = {
    "chat_history": [],
    "vector_db": None,
    "authenticated": False,
    "users": {"admin": "admin123"}
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -------------------- Gemini Setup --------------------
def get_llm():
    api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not api_key:
        st.error("Google API key missing")
        st.stop()

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.3
    )

# -------------------- BLIP Load --------------------
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, blip_model = load_blip()

def describe_image(image):
    inputs = processor(image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

# -------------------- Auth UI --------------------
def login_ui():
    st.title("🔐 SlideSense Login")

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")

        if st.button("Login"):
            if u in st.session_state.users and st.session_state.users[u] == p:
                st.session_state.authenticated = True
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        nu = st.text_input("New Username")
        np = st.text_input("New Password", type="password")

        if st.button("Create Account"):
            if nu in st.session_state.users:
                st.warning("User already exists")
            else:
                st.session_state.users[nu] = np
                st.success("Account created")

# -------------------- Auth Check --------------------
if not st.session_state.authenticated:
    login_ui()
    st.stop()

# -------------------- Sidebar --------------------
st.sidebar.success("Logged in")
if st.sidebar.button("Logout"):
    for k in defaults:
        st.session_state[k] = defaults[k]
    st.rerun()

page = st.sidebar.radio("Mode", ["📘 PDF Analyzer", "🖼 Image Recognition"])

st.title("📘 SlideSense AI Platform")

# -------------------- PDF Analyzer --------------------
if page == "📘 PDF Analyzer":

    pdf = st.file_uploader("Upload PDF", type="pdf")

    if pdf:
        if st.session_state.vector_db is None:

            reader = PdfReader(pdf)
            text = ""

            for p in reader.pages:
                if p.extract_text():
                    text += p.extract_text() + "\n"

            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
            chunks = splitter.split_text(text)

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            st.session_state.vector_db = FAISS.from_texts(chunks, embeddings)

        st.success("PDF processed successfully")

        q = st.text_input("Ask question from PDF")

        if q:
            docs = st.session_state.vector_db.similarity_search(q, k=5)
            llm = get_llm()

            history = ""
            for x, y in st.session_state.chat_history[-5:]:
                history += f"Q:{x}\nA:{y}\n"

            prompt = ChatPromptTemplate.from_template("""
History:
{history}

Context:
{context}

Question:
{question}

Rules:
- Answer only from document
- If not found say: Information not found in the document
""")

            chain = create_stuff_documents_chain(llm, prompt)
            res = chain.invoke({"context": docs, "question": q, "history": history})

            st.session_state.chat_history.append((q, res))

    st.subheader("Conversation")

    for q, a in st.session_state.chat_history:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**AI:** {a}")
        st.divider()

# -------------------- Image Recognition --------------------
if page == "🖼 Image Recognition":

    img_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if img_file:
        img = Image.open(img_file)
        st.image(img, use_column_width=True)

        desc = describe_image(img)
        st.success(desc)
