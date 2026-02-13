import streamlit as st
import time
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# -------------------- Page Config --------------------
st.set_page_config(page_title="SlideSense", page_icon="📘", layout="wide")

# -------------------- Simple Animations --------------------
def pdf_animation():
    box = st.empty()
    steps = [
        "📘 Opening document...",
        "🧠 Reading pages...",
        "📊 Preparing knowledge...",
        "✅ Ready!"
    ]
    for step in steps:
        box.info(step)
        time.sleep(0.5)

def image_animation():
    box = st.empty()
    steps = [
        "🖼 Loading image...",
        "👁 Detecting objects...",
        "🤖 Generating description...",
        "✅ Ready!"
    ]
    for step in steps:
        box.info(step)
        time.sleep(0.5)

# -------------------- Session --------------------
defaults = {
    "chat_history": [],
    "vector_db": None,
    "authenticated": False,
    "users": {"admin": "admin123"}
}

for k,v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -------------------- Auth UI --------------------
def login_ui():
    st.title("🔐 SlideSense Login")

    tab1, tab2 = st.tabs(["Login","Sign Up"])

    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")

        if st.button("Login"):
            if u in st.session_state.users and st.session_state.users[u] == p:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        nu = st.text_input("New Username")
        np = st.text_input("New Password", type="password")

        if st.button("Create Account"):
            if nu in st.session_state.users:
                st.warning("User exists")
            else:
                st.session_state.users[nu] = np
                st.success("Account created")

# -------------------- BLIP Loader --------------------
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, blip_model = load_blip()

def describe_image(img):
    inputs = processor(img, return_tensors="pt")
    out = blip_model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

# -------------------- LLM --------------------
def get_llm():
    api_key = st.secrets.get("GOOGLE_API_KEY")
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key
    )

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

page = st.sidebar.radio("Mode", ["📘 PDF Analyzer","🖼 Image Recognition"])

# =====================================================
# PDF ANALYZER
# =====================================================
if page == "📘 PDF Analyzer":

    pdf_animation()  # ✅ simple animation

    pdf = st.file_uploader("Browse PDF", type="pdf")

    if pdf:

        if st.session_state.vector_db is None:
            reader = PdfReader(pdf)
            text = ""

            for p in reader.pages:
                content = p.extract_text()
                if content:
                    text += content + "\n"

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=80
            )

            chunks = splitter.split_text(text)

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            st.session_state.vector_db = FAISS.from_texts(chunks, embeddings)

        st.success("PDF Ready")

        q = st.text_input("Ask question from PDF")

        if q:
            llm = get_llm()

            retriever = st.session_state.vector_db.as_retriever(
                search_kwargs={"k":5}
            )

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type="stuff"
            )

            res = qa_chain.invoke(q)
            answer = res["result"]

            st.session_state.chat_history.append((q,answer))

        st.subheader("💬 Conversation")

        for q,a in st.session_state.chat_history:
            st.markdown(f"🧑 {q}")
            st.markdown(f"🤖 {a}")
            st.divider()

# =====================================================
# IMAGE RECOGNITION
# =====================================================
if page == "🖼 Image Recognition":

    image_animation()  # ✅ simple animation

    img_file = st.file_uploader("Browse Image", type=["png","jpg","jpeg"])

    if img_file:
        img = Image.open(img_file)
        st.image(img, use_column_width=True)

        desc = describe_image(img)
        st.success(desc)
