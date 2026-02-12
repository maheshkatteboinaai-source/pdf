import streamlit as st
from streamlit_lottie import st_lottie
import requests
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import tempfile
import os
import pytesseract
from cohere import Client as CohereClient

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="SlideSense", page_icon="📘", layout="wide")

# -------------------- LOTTIE --------------------
def load_lottie(url: str):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        data = r.json()
        if "v" not in data:
            return None
        return data
    except:
        return None

def safe_lottie(anim, height=200):
    if isinstance(anim, dict):
        st_lottie(anim, height=height)

login_anim = load_lottie("https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json")
pdf_anim   = load_lottie("https://assets9.lottiefiles.com/packages/lf20_q5pk6p1k.json")
image_anim = load_lottie("https://assets2.lottiefiles.com/packages/lf20_iorpbol0.json")

# -------------------- SESSION DEFAULTS --------------------
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

# -------------------- AUTH UI --------------------
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
                    st.warning("User already exists")
                else:
                    st.session_state.users[nu] = np
                    st.success("Account created 🎉")

# -------------------- AUTH CHECK --------------------
if not st.session_state.authenticated:
    login_ui()
    st.stop()

# -------------------- SIDEBAR --------------------
st.sidebar.success("Logged in ✅")

if st.sidebar.button("Logout"):
    for k, v in defaults.items():
        st.session_state[k] = v
    st.rerun()

page = st.sidebar.radio("Mode", ["📘 PDF Analyzer", "🖼 Image Recognition"])

# -------------------- GEMINI LLM --------------------
def get_llm():
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("Missing GOOGLE_API_KEY in secrets.")
        st.stop()
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
    )

# -------------------- COHERE --------------------
def get_cohere_client():
    api_key = st.secrets.get("COHERE_API_KEY")
    if not api_key:
        st.error("Missing COHERE_API_KEY in secrets.")
        st.stop()
    return CohereClient(api_key=api_key, timeout=60)

def generate_text_with_cohere(prompt):
    cohere_client = get_cohere_client()
    response = cohere_client.generate(
        model="command-xlarge-nightly",
        prompt=prompt,
        temperature=0.7,
        max_tokens=800,
    )
    return response.generations[0].text.strip()

# -------------------- CACHE MODELS --------------------
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

# ========================= PDF ANALYZER =========================
if page == "📘 PDF Analyzer":

    col1, col2 = st.columns([1, 2])
    with col1:
        safe_lottie(pdf_anim, 200)

    with col2:
        st.markdown("## 📘 PDF Analyzer")
        st.markdown("Upload your document and ask AI questions from it.")

    st.divider()

    pdf = st.file_uploader("Upload PDF", type="pdf")

    if pdf:
        if st.session_state.current_pdf != pdf.name:
            st.session_state.vector_db = None
            st.session_state.current_pdf = pdf.name

        if st.session_state.vector_db is None:
            with st.spinner("Processing document..."):
                reader = PdfReader(pdf)
                text = ""
                for page_obj in reader.pages:
                    content = page_obj.extract_text()
                    if content:
                        text += content + "\n"

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=80
                )
                chunks = splitter.split_text(text)

                embeddings = load_embeddings()
                st.session_state.vector_db = FAISS.from_texts(chunks, embeddings)

        st.success("PDF Ready 🚀")

        q = st.text_input("Ask your question")

        if q:
            with st.spinner("Generating answer..."):
                docs = st.session_state.vector_db.similarity_search(q, k=5)
                llm = get_llm()

                history = "\n".join(
                    f"Q:{x}\nA:{y}" for x, y in st.session_state.chat_history[-5:]
                )

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
                res = chain.invoke({
                    "context": docs,
                    "question": q,
                    "history": history
                })

                st.session_state.chat_history.append((q, res))

        st.markdown("## 💬 AI Conversation")
        for q, a in st.session_state.chat_history:
            st.markdown(f"🧑 **You:** {q}")
            st.markdown(f"🤖 **AI:** {a}")
            st.divider()

# ========================= IMAGE RECOGNITION =========================
if page == "🖼 Image Recognition":

    col1, col2 = st.columns([1, 2])
    with col1:
        safe_lottie(image_anim, 200)

    with col2:
        st.markdown("## 🖼 VisionText Image QA")

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file:

        temp_dir = tempfile.TemporaryDirectory()
        temp_path = os.path.join(temp_dir.name, uploaded_file.name)

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        st.image(temp_path, use_container_width=True)

        # --- IMAGE CLASSIFICATION ---
        def classify_image(image_path):
            text = pytesseract.image_to_string(image_path)
            if len(text.strip()) > 30:
                return "text"
            return "natural"

        def extract_text(image_path):
            return pytesseract.image_to_string(Image.open(image_path)).strip()

        def describe_image(image_path):
            processor, model = load_blip()
            image = Image.open(image_path).convert("RGB")
            inputs = processor(image, return_tensors="pt")
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=50)
            return processor.decode(output[0], skip_special_tokens=True)

        with st.spinner("Analyzing image..."):
            image_type = classify_image(temp_path)

        if image_type == "text":
            extracted_content = extract_text(temp_path)
            st.success("Text-heavy image detected!")
        else:
            extracted_content = describe_image(temp_path)
            st.success("Natural image detected!")

        st.divider()

        question = st.text_input("Ask a question about the image:")

        if question:
            qa_prompt = f"""
Analyze the content below and answer the question.
If unclear, respond with "I don't know."

Content:
{extracted_content}

Question:
{question}

Answer:
"""
            with st.spinner("Generating answer..."):
                answer = generate_text_with_cohere(qa_prompt)

            st.markdown("### 🤖 AI Answer")
            st.success(answer)
