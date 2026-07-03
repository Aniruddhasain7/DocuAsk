import os
import tempfile
import warnings
import logging
import pathlib
import hashlib
from dotenv import load_dotenv
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


warnings.filterwarnings("ignore")
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

load_dotenv()

def load_document(file_path):
    ext = file_path.split(".")[-1].lower()

    if ext == "pdf":
        loader = PyPDFLoader(file_path)
    elif ext in ["docx", "doc"]:
        loader = Docx2txtLoader(file_path)
    elif ext == "txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext == "csv":
        loader = CSVLoader(file_path, encoding="utf-8")
    else:
        raise ValueError("Unsupported file format")

    return loader.load()


def setup_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)
    return FAISS.from_documents(chunks, embeddings)


def create_chain(vectorstore):
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        verbose=False
    )

st.set_page_config(
    page_title="DocuAsk",
    page_icon="📑",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

section[data-testid="stSidebar"] {
    background-color: #121218 !important;
    border-right: 1px solid #23232f;
}

.doc-card {
    background: #1C1C26;
    border: 1px solid #2D2D3E;
    border-radius: 12px;
    padding: 16px;
    margin: 12px 0;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}
.doc-card-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
}
.doc-icon {
    font-size: 1.5rem;
}
.doc-title {
    font-weight: 600;
    color: #FFFFFF;
    word-break: break-all;
    font-size: 0.95rem;
}
.doc-card-body {
    font-size: 0.85rem;
    color: #A0AEC0;
}
.doc-meta {
    margin-bottom: 4px;
}
.status-badge {
    display: inline-block;
    background-color: rgba(108, 92, 231, 0.15);
    color: #a29bfe;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
    border: 1px solid rgba(108, 92, 231, 0.3);
    margin-top: 8px;
}

.stButton>button[kind="primary"] {
    background: linear-gradient(135deg, #6C5CE7 0%, #4834DF 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 12px rgba(108, 92, 231, 0.25) !important;
    transition: all 0.3s ease !important;
}
.stButton>button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(108, 92, 231, 0.4) !important;
    background: linear-gradient(135deg, #8172eb 0%, #5f4de6 100%) !important;
}

.stButton>button[kind="secondary"] {
    background-color: transparent !important;
    color: #E2E8F0 !important;
    border: 1px solid #3A3A4A !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
}
.stButton>button[kind="secondary"]:hover {
    border-color: #6C5CE7 !important;
    color: #6C5CE7 !important;
    background-color: rgba(108, 92, 231, 0.05) !important;
}

.header-container {
    text-align: center;
    margin-bottom: 30px;
    padding: 20px 0;
}
.header-logo {
    font-size: 3rem;
    margin-bottom: 10px;
    animation: float 3s ease-in-out infinite;
}
.header-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a29bfe 0%, #6C5CE7 50%, #4834DF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -0.5px;
}
.header-tagline {
    color: #718096;
    font-size: 1.1rem;
    margin-top: 8px;
    font-weight: 400;
}
@keyframes float {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
    100% { transform: translateY(0px); }
}

.welcome-card {
    background: rgba(28, 28, 38, 0.6);
    border: 1px solid #23232f;
    border-radius: 16px;
    padding: 30px;
    max-width: 600px;
    margin: 40px auto;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    backdrop-filter: blur(4px);
}
.welcome-card h3 {
    color: #FFFFFF;
    margin-top: 0;
    margin-bottom: 24px;
    font-size: 1.4rem;
    font-weight: 600;
    text-align: center;
}
.step-item {
    display: flex;
    align-items: flex-start;
    gap: 16px;
    margin-bottom: 20px;
}
.step-item:last-child {
    margin-bottom: 0;
}
.step-number {
    background: linear-gradient(135deg, #6C5CE7, #4834DF);
    color: white;
    font-weight: 700;
    font-size: 0.95rem;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    box-shadow: 0 4px 10px rgba(108, 92, 231, 0.3);
}
.step-text {
    flex-grow: 1;
}
.step-text strong {
    color: #E2E8F0;
    font-size: 1.05rem;
}
.step-text p {
    color: #A0AEC0;
    margin: 4px 0 0 0;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-container">
    <div class="header-logo">📑</div>
    <div class="header-title">DocuAsk</div>
    <div class="header-tagline">AI-powered Document Intelligence</div>
</div>
""", unsafe_allow_html=True)

MIME_TO_EXT = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/msword": ".doc",
    "text/plain": ".txt",
    "text/csv": ".csv",
    "application/csv": ".csv"
}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None

if "processed_file_hash" not in st.session_state:
    st.session_state.processed_file_hash = None

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

def reset_session():
    st.session_state.chat_history = []
    st.session_state.conversation_chain = None
    st.session_state.processed_file_hash = None
    if "vectorstore" in st.session_state:
        del st.session_state.vectorstore
    st.session_state.uploader_key += 1

def format_size(bytes_size):
    if bytes_size < 1024:
        return f"{bytes_size} B"
    elif bytes_size < 1024 * 1024:
        return f"{bytes_size / 1024:.1f} KB"
    else:
        return f"{bytes_size / (1024 * 1024):.1f} MB"

with st.sidebar:
    st.markdown('<div style="font-size: 1.3rem; font-weight: 700; color: #FFFFFF; display: flex; align-items: center; gap: 8px;">📊 Dashboard</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### 📥 1. Upload Document")
    uploaded_file = st.file_uploader(
        "Upload a document to analyze",
        type=["pdf", "docx", "txt", "csv"],
        key=f"file_uploader_{st.session_state.uploader_key}",
        label_visibility="collapsed"
    )
    
    if uploaded_file and st.session_state.conversation_chain:
        file_ext_display = pathlib.Path(uploaded_file.name).suffix.lower().replace(".", "").upper()
        file_size = format_size(uploaded_file.size)
        st.markdown(f"""
        <div class="doc-card">
            <div class="doc-card-header">
                <span class="doc-icon">📄</span>
                <span class="doc-title">{uploaded_file.name}</span>
            </div>
            <div class="doc-card-body">
                <div class="doc-meta"><strong>Format:</strong> {file_ext_display}</div>
                <div class="doc-meta"><strong>Size:</strong> {file_size}</div>
                <span class="status-badge">⚡ Processed & Ready</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif not uploaded_file:
        st.markdown("""
        <div style='background-color: rgba(255,255,255,0.02); border: 1px dashed rgba(255,255,255,0.1); border-radius: 8px; padding: 16px; text-align: center; color: #718096; font-size: 0.9rem;'>
            No active document. Upload a file above to begin.
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    if uploaded_file or len(st.session_state.chat_history) > 0:
        st.markdown("### ⚙️ 2. Session Actions")
        if st.button("Reset Session", type="secondary", use_container_width=True):
            reset_session()
            st.rerun()
        st.markdown("---")

    st.markdown("### ℹ️ Supported Formats")
    st.markdown("""
    - 📑 **PDF** (`.pdf`)
    - 📝 **Word** (`.docx`, `.doc`)
    - 📄 **Text** (`.txt`)
    - 📊 **Data** (`.csv`)
    """)

if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    if st.session_state.processed_file_hash != file_hash:
        file_ext = pathlib.Path(uploaded_file.name).suffix.lower()
        if file_ext not in [".pdf", ".docx", ".doc", ".txt", ".csv"]:
            mime_type = getattr(uploaded_file, "type", "")
            if mime_type in MIME_TO_EXT:
                file_ext = MIME_TO_EXT[mime_type]
            else:
                st.error("Unsupported file format. Please upload a PDF, DOCX, TXT, or CSV file.")
                st.stop()

        temp_path = None
        success = False
        try:
            with st.spinner("Processing document..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                    tmp.write(file_bytes)
                    temp_path = tmp.name

                documents = load_document(temp_path)
                vectorstore = setup_vectorstore(documents)
                st.session_state.vectorstore = vectorstore
                st.session_state.conversation_chain = create_chain(vectorstore)
                st.session_state.chat_history = []
                st.session_state.processed_file_hash = file_hash
                success = True
        except Exception as e:
            st.error(f"Failed to process document: {e}")
            logging.error(f"Error loading document: {e}", exc_info=True)
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as cleanup_err:
                    logging.warning(f"Failed to clean up temporary file {temp_path}: {cleanup_err}")

        if success:
            st.success("Document processed successfully!")
            st.rerun()

if not st.session_state.conversation_chain:
    st.markdown("""
    <div class="welcome-card">
        <h3>Get Started in Seconds</h3>
        <div class="step-item">
            <span class="step-number">1</span>
            <div class="step-text">
                <strong>Upload a Document</strong>
                <p>Drag and drop a PDF, Word document, Text file, or CSV into the sidebar file uploader.</p>
            </div>
        </div>
        <div class="step-item">
            <span class="step-number">2</span>
            <div class="step-text">
                <strong>Ask Questions</strong>
                <p>Use the chat box at the bottom to query information or request summaries.</p>
            </div>
        </div>
        <div class="step-item">
            <span class="step-number">3</span>
            <div class="step-text">
                <strong>Get Instant Answers</strong>
                <p>Retrieve context-aware responses instantly, backed by your loaded source material.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

user_input = st.chat_input("Ask something about your document...")

if user_input:
    if not st.session_state.conversation_chain:
        st.warning("Please upload a document first.")
    else:
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            response = st.session_state.conversation_chain.invoke({
                "question": user_input
            })

            answer = response["answer"]
            st.markdown(answer)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer
            })