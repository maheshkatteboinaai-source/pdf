import streamlit as st
from streamlit_lottie import st_lottie
import requests
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import tempfile
import os
import pytesseract

# Fix Tesseract path (Streamlit Cloud)
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="SlideSense", page_icon="📘", layout="wide")

# ---------------- LOTTIE ----------------
def load_lottie(url):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

def safe_lottie(anim, height=200):
    if isinstance(anim, dict):
        st_lottie(anim, height=height)

login_anim = load_lottie("https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json")
pdf_anim   = load_lottie("https://assets9.lottiefiles.com/packages/lf20_q5pk6p1k.json")
image_anim = load_lottie("https://assets2.lottiefiles.com/packages/lf20_iorpbol0.json")

# ---------------- SESSION DEFAULTS ----------------
defaults = {
    "chat_history": [],
    "vector_db": None,
    "authenticated": False,
    "users": {"admin": "admin123"},
    "current_pdf": None,
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------- AUTH ----------------
def login_ui():
    col1, col2 = st.columns([1, 1])

    with col1:
        safe_lottie(login_anim, 300)

    with col2:
        st.markdown("## 🔐 Welcome to SlideSense")
        tab1, tab2 = st.tabs(["Login", "Sign Up"])

        with tab1:
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.button("Login"):
                if u in st.session_state.users and st.session_state.users[u] == p:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid credentials ❌")

        with tab2:
            nu = st.text_input("New Username")
            np = st.text_input("New Password", type="password")
            if st.button("Create Account"):
                if nu in st.session_state.users:
                    st.warning("User exists")
                else:
                    st.session_state.users[nu] = np
                    st.success("Account created 🎉")

if not st.session_state.authenticated:
    login_ui()
    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.success("Logged in ✅")

if st.sidebar.button("Logout"):
    for k, v in defaults.items():
        st.session_state[k] = v
    st.rerun()

page = st.sidebar.radio("Mode", ["📘 PDF Analyzer", "🖼 Image Recognition"])

# ---------------- GEMINI ----------------
def get_llm():
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("Missing GOOGLE_API_KEY")
        st.stop()

    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
    )

# ---------------- CACHE MODELS ----------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.eval()
    return processor, model

# ================= PDF ANALYZER =================
if page == "📘 PDF Analyzer":

    st.markdown("## 📘 PDF Analyzer")

    pdf = st.file_uploader("Upload PDF", type="pdf")

    if pdf:

        if st.session_state.current_pdf != pdf.name:
            st.session_state.vector_db = None
            st.session_state.current_pdf = pdf.name

        if st.session_state.vector_db is None:
            with st.spinner("Processing PDF..."):

                reader = PdfReader(pdf)
                text = ""

                for page_obj in reader.pages:
                    content = page_obj.extract_text()
                    if content:
                        text += str(content) + "\n"

                if not text.strip():
                    st.error("No readable text found.")
                    st.stop()

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=80
                )

                chunks = splitter.split_text(text)

                embeddings = load_embeddings()
                st.session_state.vector_db = FAISS.from_texts(chunks, embeddings)

        st.success("PDF Ready 🚀")

        q = st.text_input("Ask a question")

        if q:
            with st.spinner("Generating answer..."):
                llm = get_llm()
                retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 5})

                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=retriever,
                    chain_type="stuff"
                )

                result = qa_chain.invoke({"query": q})
                answer = result.get("result") or result.get("answer")

                st.session_state.chat_history.append((q, answer))

        st.markdown("## 💬 Conversation")

        for q, a in st.session_state.chat_history:
            st.markdown(f"🧑 **You:** {q}")
            st.markdown(f"🤖 **AI:** {a}")
            st.divider()

# ================= IMAGE RECOGNITION =================
if page == "🖼 Image Recognition":

    st.markdown("## 🖼 VisionText Image QA")

    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if uploaded_file:

        with tempfile.TemporaryDirectory() as temp_dir:

            temp_path = os.path.join(temp_dir, uploaded_file.name)

            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            st.image(temp_path, use_container_width=True)

            def classify_image(path):
                try:
                    text = pytesseract.image_to_string(path)
                    return "text" if len(text.strip()) > 30 else "natural"
                except:
                    return "natural"

            def extract_text(path):
                return pytesseract.image_to_string(Image.open(path)).strip()

            def describe_image(path):
                processor, model = load_blip()
                image = Image.open(path).convert("RGB")
                inputs = processor(image, return_tensors="pt")
                with torch.no_grad():
                    output = model.generate(**inputs, max_new_tokens=50)
                return processor.decode(output[0], skip_special_tokens=True)

            with st.spinner("Analyzing image..."):
                image_type = classify_image(temp_path)

            if image_type == "text":
                extracted_content = extract_text(temp_path)
            else:
                extracted_content = describe_image(temp_path)

            question = st.text_input("Ask about the image")

            if question:
                llm = get_llm()

                prompt = f"""
Content:
{extracted_content}

Question:
{question}

Answer:
"""

                with st.spinner("Generating answer..."):
                    answer = llm.invoke(prompt).content

                st.success(answer)
