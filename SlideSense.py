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
import asyncio
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# -------------------- Page Config --------------------
st.set_page_config(page_title="SlideSense", page_icon="📘", layout="wide")

# -------------------- Lottie Loader --------------------
def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Animations
login_anim = load_lottie("https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json")
pdf_anim = load_lottie("https://assets9.lottiefiles.com/packages/lf20_q5pk6p1k.json")
image_anim = load_lottie("https://assets2.lottiefiles.com/packages/lf20_iorpbol0.json")

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

# -------------------- Auth UI --------------------
def login_ui():
    col1, col2 = st.columns([1, 1])

    with col1:
        st_lottie(login_anim, height=300)

    with col2:
        st.markdown("## 🔐 Welcome to SlideSense")
        st.markdown("### AI Powered Learning Platform")

        tab1, tab2 = st.tabs(["Login", "Sign Up"])

        with tab1:
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.button("Login"):
                if u in st.session_state.users and st.session_state.users[u] == p:
                    st.success("Login Successful 🚀")
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

# -------------------- BLIP --------------------
@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, blip_model = load_blip()

def describe_image(image):
    inputs = processor(image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

# -------------------- Auth Check --------------------
if not st.session_state.authenticated:
    login_ui()
    st.stop()

# -------------------- Sidebar --------------------
st.sidebar.success("Logged in ✅")
if st.sidebar.button("Logout"):
    for k in defaults:
        st.session_state[k] = defaults[k]
    st.rerun()

page = st.sidebar.radio("Mode", ["📘 PDF Analyzer", "🖼 Image Recognition"])

st.sidebar.markdown("### 💬 History")
for q, a in st.session_state.chat_history[-8:]:
    st.sidebar.markdown(f"- {q[:30]}")

# ========================= PDF ANALYZER =========================
if page == "📘 PDF Analyzer":

    col1, col2 = st.columns([1, 2])

    with col1:
        st_lottie(pdf_anim, height=200)

    with col2:
        st.markdown("## 📘 PDF Analyzer")
        st.markdown("Upload your document and ask AI questions from it.")

    st.divider()

    pdf = st.file_uploader("Upload PDF", type="pdf")

    if pdf:
        if st.session_state.vector_db is None:
            with st.spinner("🧠 Processing your document..."):

                reader = PdfReader(pdf)
                text = ""
                for p in reader.pages:
                    if p.extract_text():
                        text += p.extract_text() + "\n"

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=80
                )
                chunks = splitter.split_text(text)

                try:
                    asyncio.get_running_loop()
                except:
                    asyncio.set_event_loop(asyncio.new_event_loop())

                embeddings = HuggingFaceEmbeddings(
                    model_name='sentence-transformers/all-MiniLM-L6-v2'
                )

                st.session_state.vector_db = FAISS.from_texts(chunks, embeddings)

        st.success("PDF Ready 🚀")

        q = st.text_input("Ask your question")

        if q:
            with st.spinner("🤖 Generating answer..."):

                docs = st.session_state.vector_db.similarity_search(q, k=5)

                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

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
        st_lottie(image_anim, height=200)

    with col2:
        st.markdown("## 🖼 Image Recognition")
        st.markdown("Upload an image, get AI description, and ask questions about it.")

    st.divider()

    img_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if img_file:
        img = Image.open(img_file)
        st.image(img, use_column_width=True)

        # ---------- IMAGE DESCRIPTION ----------
        with st.spinner("🤖 Analyzing image..."):
            short_desc = describe_image(img)

            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
            prompt = f"""
            The image caption is: "{short_desc}"

            Generate a detailed description of the image in at least 4 meaningful lines.
            Describe:
            - Objects
            - Scene
            - Environment
            - Possible context or activity
            """
            detailed_desc = llm.invoke(prompt)

        st.markdown("### 📝 AI Image Description")
        st.success(detailed_desc.content)

        st.divider()

        # ---------- QUESTION ANSWER SECTION ----------
        st.markdown("### 💬 Ask Questions About the Image")

        img_question = st.text_input("Ask a question about this image:")

        if img_question:
            with st.spinner("🤖 Thinking..."):
                qa_prompt = f"""
                Image description:
                {detailed_desc.content}

                Question:
                {img_question}

                Answer clearly based only on the image description.
                """

                answer = llm.invoke(qa_prompt)

            st.markdown("#### 🤖 AI Answer")
            st.success(answer.content)
