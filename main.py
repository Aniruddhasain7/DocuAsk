import os
import tempfile
import warnings
import logging
import pathlib
import hashlib
import zipfile
import base64
import io
import uuid

os.environ["ANONYMIZED_TELEMETRY"] = "False"
from dotenv import load_dotenv
import streamlit as st
from PIL import Image

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import chromadb
from chromadb.api.client import SharedSystemClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

warnings.filterwarnings("ignore")
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

load_dotenv()

def explain_image_with_groq(pil_img, label="embedded image"):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return ""
    try:
        from groq import Groq
        client = Groq(api_key=api_key)

        if pil_img.mode in ("RGBA", "LA") or (pil_img.mode == "P" and "transparency" in pil_img.info):
            bg = Image.new("RGB", pil_img.size, (255, 255, 255))
            bg.paste(pil_img, mask=pil_img.split()[-1])
            pil_img = bg
        else:
            pil_img = pil_img.convert("RGB")

        max_dimension = 1200
        if max(pil_img.size) > max_dimension:
            pil_img = pil_img.copy()
            pil_img.thumbnail((max_dimension, max_dimension))

        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=85)
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{b64_str}"

        prompt = (
            "Analyze this diagram/image found in a document in comprehensive detail.\n"
            "1. Overview: What is this diagram, flowchart, UML chart, UI wireframe, or visual illustrating?\n"
            "2. Complete Entities & Text: Transcribe all class names, method names, variable names, labels, arrows, relationships, and data values.\n"
            "3. Structural Relationships: Detail connections, hierarchies, inheritance, and dependencies shown in the diagram.\n"
            "Explain everything factually and thoroughly so someone reading this text gets all the information contained in the visual diagram."
        )

        response = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            }],
            temperature=0.1,
            max_tokens=1500
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        logging.warning(f"Groq Vision analysis error for {label}: {e}")
        return ""


def load_document(file_path):
    ext = file_path.split(".")[-1].lower()

    if ext == "pdf":
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        try:
            import pypdfium2 as pdfium
            with pdfium.PdfDocument(file_path) as pdf:
                for idx in range(min(len(pdf), len(docs))):
                    doc = docs[idx]
                    if len(doc.page_content.strip()) < 250 or len(pdf) <= 3:
                        pil_img = pdf[idx].render(scale=2).to_pil()
                        explanation = explain_image_with_groq(pil_img, label=f"PDF page {idx + 1}")
                        if explanation.strip():
                            doc.page_content += f"\n\n[Visual Diagram Analysis for Page {idx + 1}]:\n{explanation}"
        except Exception as e:
            logging.warning(f"PDF visual inspection error: {e}")

        return docs

    elif ext in ["docx", "doc"]:
        loader = Docx2txtLoader(file_path)
        docs = loader.load()

        if ext == "docx" and docs:
            try:
                with zipfile.ZipFile(file_path, "r") as docx_zip:
                    image_files = [f for f in docx_zip.namelist() if f.startswith("word/media/")]
                    explanations = []
                    for img_name in image_files:
                        img_data = docx_zip.read(img_name)
                        try:
                            pil_img = Image.open(io.BytesIO(img_data))
                            if pil_img.width >= 100 and pil_img.height >= 100:
                                explanation = explain_image_with_groq(
                                    pil_img,
                                    label=f"DOCX embedded image {img_name}"
                                )
                                if explanation.strip():
                                    explanations.append(
                                        f"\n\n[Detected Embedded Figure ({os.path.basename(img_name)}) Visual Explanation]:\n"
                                        f"{explanation}"
                                    )
                        except Exception as img_err:
                            logging.warning(f"Failed to process DOCX image {img_name}: {img_err}")

                    if explanations:
                        docs[0].page_content += "".join(explanations)
            except Exception as docx_err:
                logging.warning(f"DOCX image inspection error: {docx_err}")

        return docs

    elif ext == "txt":
        loader = TextLoader(file_path, encoding="utf-8")
        return loader.load()

    elif ext == "csv":
        loader = CSVLoader(file_path, encoding="utf-8")
        return loader.load()

    else:
        raise ValueError("Unsupported file format")


def setup_vectorstore(documents):
    try:
        SharedSystemClient.clear_system_cache()
    except Exception:
        pass

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("No readable text or visual content could be extracted from this document.")

    client = chromadb.Client()
    collection_name = f"docuask_{uuid.uuid4().hex[:12]}"
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=client,
        collection_name=collection_name
    )
    total_chars = sum(len(d.page_content) for d in documents)
    return vectorstore, len(chunks), total_chars


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


icon_path = os.path.join(os.path.dirname(__file__), "assets", "icon.png")
page_icon_val = icon_path if os.path.exists(icon_path) else "🤖"

st.set_page_config(
    page_title="DocuAsk",
    page_icon=page_icon_val,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Orbitron:wght@500;600;700;800;900&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --cyan-glow: #00F2FE;
    --neon-blue: #4FACFE;
    --cyber-purple: #7000FF;
    --neon-green: #00FF88;
    --dark-bg: #07090E;
    --card-surface: rgba(13, 18, 28, 0.75);
    --border-glow: rgba(0, 242, 254, 0.22);
    --border-dim: rgba(255, 255, 255, 0.08);
}

html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #E2E8F0;
    background-color: var(--dark-bg);
}

.stApp {
    background-color: #07090E;
    background-image: 
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(0, 242, 254, 0.09), transparent 70%),
        radial-gradient(circle at 90% 85%, rgba(112, 0, 255, 0.06), transparent 50%),
        linear-gradient(rgba(0, 242, 254, 0.02) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0, 242, 254, 0.02) 1px, transparent 1px);
    background-size: 100% 100%, 100% 100%, 36px 36px, 36px 36px;
    background-attachment: fixed;
}

header[data-testid="stHeader"] {
    background: transparent;
}

section[data-testid="stSidebar"] {
    background-color: #0A0D15 !important;
    border-right: 1px solid rgba(0, 242, 254, 0.15) !important;
    box-shadow: 4px 0 24px rgba(0, 0, 0, 0.5);
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}

::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #07090E;
}
::-webkit-scrollbar-thumb {
    background: rgba(0, 242, 254, 0.25);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--cyan-glow);
}

.cyber-header-wrapper {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    padding: 16px 0 12px;
    margin-bottom: 20px;
    position: relative;
}

.cyber-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    color: #00F2FE;
    background: rgba(0, 242, 254, 0.08);
    border: 1px solid rgba(0, 242, 254, 0.35);
    border-radius: 20px;
    padding: 4px 14px;
    margin-bottom: 12px;
    letter-spacing: 0.15em;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    box-shadow: 0 0 16px rgba(0, 242, 254, 0.15);
}

.cyber-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 2.8rem;
    font-weight: 900;
    letter-spacing: 2px;
    background: linear-gradient(135deg, #FFFFFF 0%, #00F2FE 55%, #7000FF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.15;
    text-shadow: 0 0 35px rgba(0, 242, 254, 0.3);
}

.cyber-subtitle {
    font-family: 'Space Grotesk', sans-serif;
    color: #94A3B8;
    font-size: 0.95rem;
    margin-top: 8px;
    letter-spacing: 0.05em;
}

.pulse-dot-green {
    width: 7px;
    height: 7px;
    background: #00FF88;
    border-radius: 50%;
    display: inline-block;
    box-shadow: 0 0 10px #00FF88;
    animation: pulse-green 2s infinite ease-in-out;
}

@keyframes pulse-green {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 255, 136, 0.7); }
    70% { transform: scale(1.1); box-shadow: 0 0 0 6px rgba(0, 255, 136, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 255, 136, 0); }
}

.hud-card {
    background: rgba(13, 19, 32, 0.7);
    border: 1px solid rgba(0, 242, 254, 0.18);
    border-radius: 12px;
    padding: 16px;
    margin: 12px 0;
    position: relative;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4), inset 0 0 12px rgba(0, 242, 254, 0.03);
    transition: all 0.3s ease;
}

.hud-card::before {
    content: "";
    position: absolute;
    top: -1px;
    left: 12px;
    width: 24px;
    height: 2px;
    background: #00F2FE;
    box-shadow: 0 0 8px #00F2FE;
}

.hud-card-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    padding-bottom: 8px;
}

.hud-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 700;
    color: #00F2FE;
    background: rgba(0, 242, 254, 0.1);
    padding: 2px 8px;
    border-radius: 4px;
    border: 1px solid rgba(0, 242, 254, 0.25);
    letter-spacing: 0.05em;
}

.hud-filename {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    color: #F8FAFC;
    font-size: 0.95rem;
    word-break: break-all;
}

.hud-stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin: 10px 0;
}

.hud-stat-box {
    background: rgba(7, 10, 18, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 6px;
    padding: 8px 10px;
}

.hud-stat-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #64748B;
    text-transform: uppercase;
}

.hud-stat-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    font-weight: 700;
    color: #38BDF8;
    margin-top: 2px;
}

.hud-status-badge {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(0, 255, 136, 0.08);
    border: 1px solid rgba(0, 255, 136, 0.25);
    color: #00FF88;
    padding: 6px 12px;
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    margin-top: 10px;
    letter-spacing: 0.05em;
}

.capability-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    max-width: 900px;
    margin: 24px auto;
}

.capability-card {
    background: rgba(13, 19, 32, 0.6);
    border: 1px solid rgba(0, 242, 254, 0.12);
    border-radius: 12px;
    padding: 20px;
    position: relative;
    backdrop-filter: blur(8px);
    transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
}

.capability-card:hover {
    transform: translateY(-3px);
    border-color: rgba(0, 242, 254, 0.35);
    box-shadow: 0 10px 28px rgba(0, 242, 254, 0.08);
}

.capability-icon {
    font-size: 1.8rem;
    margin-bottom: 12px;
    display: inline-block;
}

.capability-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #F8FAFC;
    margin-bottom: 6px;
}

.capability-desc {
    font-size: 0.85rem;
    color: #94A3B8;
    line-height: 1.45;
}

.stButton>button {
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: 0.06em !important;
    border-radius: 8px !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    font-weight: 600 !important;
}

.stButton>button[kind="secondary"] {
    background: rgba(13, 19, 32, 0.8) !important;
    border: 1px solid rgba(0, 242, 254, 0.25) !important;
    color: #E2E8F0 !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
}

.stButton>button[kind="secondary"]:hover {
    border-color: #00F2FE !important;
    color: #00F2FE !important;
    box-shadow: 0 0 16px rgba(0, 242, 254, 0.25) !important;
    transform: translateY(-1px) !important;
}

div[data-testid="stFileUploader"] {
    background: rgba(13, 19, 32, 0.6);
    border: 1px dashed rgba(0, 242, 254, 0.3);
    border-radius: 12px;
    padding: 12px;
    transition: all 0.3s ease;
}

div[data-testid="stFileUploader"]:hover {
    border-color: #00F2FE;
    box-shadow: 0 0 20px rgba(0, 242, 254, 0.15);
}

div[data-testid="stChatMessage"] {
    background: rgba(11, 16, 26, 0.65) !important;
    border: 1px solid rgba(255, 255, 255, 0.07) !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
    margin-bottom: 12px !important;
    backdrop-filter: blur(10px) !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25) !important;
}

div[data-testid="stChatInput"] {
    border-radius: 12px !important;
    border: 1px solid rgba(0, 242, 254, 0.25) !important;
    background: rgba(10, 14, 24, 0.8) !important;
    backdrop-filter: blur(12px) !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4) !important;
    transition: all 0.3s ease !important;
}

div[data-testid="stChatInput"]:focus-within {
    border-color: #00F2FE !important;
    box-shadow: 0 0 24px rgba(0, 242, 254, 0.25) !important;
}

.format-chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 8px;
}

.format-chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.09);
    border-radius: 6px;
    padding: 4px 10px;
    color: #94A3B8;
    display: flex;
    align-items: center;
    gap: 6px;
}

.format-chip strong {
    color: #00F2FE;
}

.chat-role-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    margin-bottom: 6px;
    display: inline-block;
}

.role-user {
    color: #38BDF8;
}

.role-assistant {
    color: #00FF88;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="cyber-header-wrapper">
    <div class="cyber-title">DOCUASK</div>
    <div class="cyber-subtitle">High-Dimensional Vector Synthesis & Document Intelligence Interface</div>
</div>
""", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None

if "processed_file_hash" not in st.session_state:
    st.session_state.processed_file_hash = None

if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = {}

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

def reset_session():
    st.session_state.chat_history = []
    st.session_state.conversation_chain = None
    st.session_state.processed_file_hash = None
    st.session_state.doc_stats = {}
    if "vectorstore" in st.session_state:
        del st.session_state.vectorstore
    try:
        from chromadb.api.client import SharedSystemClient
        SharedSystemClient.clear_system_cache()
    except Exception:
        pass
    st.session_state.uploader_key += 1

def format_size(bytes_size):
    if bytes_size < 1024:
        return f"{bytes_size} B"
    elif bytes_size < 1024 * 1024:
        return f"{bytes_size / 1024:.1f} KB"
    else:
        return f"{bytes_size / (1024 * 1024):.1f} MB"

with st.sidebar:
    st.markdown("""
    <div style="margin-bottom: 12px;">
        <span style="font-family: 'Orbitron', sans-serif; font-weight: 800; font-size: 1.1rem; color: #FFFFFF; letter-spacing: 1px;">
            CONTROL PANEL
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div style="height: 1px; background: rgba(0, 242, 254, 0.15); margin: 8px 0 16px;"></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #64748B; margin-bottom: 8px; letter-spacing: 0.08em;">
        [ INGESTION PROTOCOL // UPLOAD ]
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload document to analyze",
        type=["pdf", "docx", "txt", "csv"],
        key=f"file_uploader_{st.session_state.uploader_key}",
        label_visibility="collapsed"
    )
    
    if uploaded_file and st.session_state.conversation_chain:
        file_ext_display = pathlib.Path(uploaded_file.name).suffix.lower().replace(".", "").upper()
        file_size = format_size(uploaded_file.size)
        chunks_count = st.session_state.doc_stats.get("chunks", "—")
        chars_count = st.session_state.doc_stats.get("chars", 0)
        chars_display = f"{chars_count:,}" if chars_count else "—"

        st.markdown(f"""
        <div class="hud-card">
            <div class="hud-card-header">
                <span class="hud-tag">{file_ext_display}</span>
                <span class="hud-filename">{uploaded_file.name}</span>
            </div>
            <div class="hud-stat-grid">
                <div class="hud-stat-box">
                    <div class="hud-stat-label">File Size</div>
                    <div class="hud-stat-val">{file_size}</div>
                </div>
                <div class="hud-stat-box">
                    <div class="hud-stat-label">Vector Chunks</div>
                    <div class="hud-stat-val">{chunks_count}</div>
                </div>
                <div class="hud-stat-box">
                    <div class="hud-stat-label">Raw Characters</div>
                    <div class="hud-stat-val">{chars_display}</div>
                </div>
                <div class="hud-stat-box">
                    <div class="hud-stat-label">Retriever</div>
                    <div class="hud-stat-val">CHROMA-384D</div>
                </div>
            </div>
            <div class="hud-status-badge">
                <span class="pulse-dot-green"></span>
                NEURAL INDEX LIVE & READY
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif not uploaded_file:
        st.markdown("""
        <div style='background-color: rgba(0, 242, 254, 0.03); border: 1px dashed rgba(0, 242, 254, 0.2); border-radius: 8px; padding: 14px; text-align: center; color: #64748B; font-family: "JetBrains Mono", monospace; font-size: 0.78rem;'>
            STANDBY: AWAITING DOCUMENT FEED
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown('<div style="height: 1px; background: rgba(0, 242, 254, 0.12); margin: 18px 0;"></div>', unsafe_allow_html=True)
    
    if uploaded_file or len(st.session_state.chat_history) > 0:
        st.markdown("""
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #64748B; margin-bottom: 8px; letter-spacing: 0.08em;">
            [ SESSION CONTROLS ]
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔄 CLEAR & RESET SESSION", type="secondary", use_container_width=True):
            reset_session()
            st.rerun()
        st.markdown('<div style="height: 1px; background: rgba(0, 242, 254, 0.12); margin: 18px 0;"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #64748B; margin-bottom: 8px; letter-spacing: 0.08em;">
        [ SUPPORTED INGEST MODULES ]
    </div>
    <div class="format-chip-row">
        <div class="format-chip"><strong>PDF</strong> Smart Visual & Text</div>
        <div class="format-chip"><strong>DOCX</strong> Word & Embedded Graphics</div>
        <div class="format-chip"><strong>TXT</strong> Raw Stream</div>
        <div class="format-chip"><strong>CSV</strong> Tabular Matrix</div>
    </div>
    """, unsafe_allow_html=True)

if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    if st.session_state.processed_file_hash != file_hash:
        file_ext = pathlib.Path(uploaded_file.name).suffix.lower()
        if file_ext not in [".pdf", ".docx", ".doc", ".txt", ".csv"]:
            st.error("Unsupported file format. Please upload a PDF, DOCX, TXT, or CSV file.")
            st.stop()

        temp_path = None
        success = False
        try:
            with st.spinner("🤖 Compiling Neural Index & Embedding Chunks..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                    tmp.write(file_bytes)
                    temp_path = tmp.name

                documents = load_document(temp_path)
                vectorstore, chunks_count, total_chars = setup_vectorstore(documents)
                st.session_state.vectorstore = vectorstore
                st.session_state.conversation_chain = create_chain(vectorstore)
                st.session_state.chat_history = []
                st.session_state.processed_file_hash = file_hash
                st.session_state.doc_stats = {
                    "chunks": chunks_count,
                    "chars": total_chars
                }
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
            st.toast("⚡ Neural Vector Index Compiled Successfully!", icon="🤖")
            st.rerun()

if not st.session_state.conversation_chain:
    st.markdown("""
    <div class="capability-grid">
        <div class="capability-card">
            <div class="capability-icon">⚡</div>
            <div class="capability-title">Parallel Embeddings</div>
            <div class="capability-desc">High-density chunking with Chroma vector similarity mapping for instantaneous sub-second recall.</div>
        </div>
        <div class="capability-card">
            <div class="capability-icon">🧠</div>
            <div class="capability-title">Zero-Hallucination RAG</div>
            <div class="capability-desc">Multi-turn contextual reasoning grounded strictly within the bounds of your ingested files.</div>
        </div>
        <div class="capability-card">
            <div class="capability-icon">🛡️</div>
            <div class="capability-title">Autonomous Parsing</div>
            <div class="capability-desc">Seamless ingestion pipeline supporting multi-modal enterprise documents: PDF, DOCX, TXT, CSV, & Visual Diagrams.</div>
        </div>
    </div>
    <div style="background: rgba(13, 19, 32, 0.6); border: 1px dashed rgba(0, 242, 254, 0.25); border-radius: 12px; padding: 24px; text-align: center; max-width: 600px; margin: 30px auto; backdrop-filter: blur(8px);">
        <div style="font-family: 'Orbitron', sans-serif; font-size: 0.95rem; color: #00F2FE; letter-spacing: 0.1em; margin-bottom: 8px;">
            AWAITING DOCUMENT FEED
        </div>
        <p style="color: #94A3B8; font-size: 0.88rem; margin: 0;">
            Upload a PDF, Word document, Text file, or CSV from the sidebar control panel to initialize the neural index.
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.chat_history:
        avatar = "👤" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            role_label = "USER" if msg["role"] == "user" else "DOCUASK AI"
            label_class = "role-user" if msg["role"] == "user" else "role-assistant"
            st.markdown(f'<div class="chat-role-label {label_class}">{role_label}</div>', unsafe_allow_html=True)
            st.markdown(msg["content"])

user_input = st.chat_input("Ask something about the document...")

if user_input:
    if not st.session_state.conversation_chain:
        st.warning("⚠️ ACCESS DENIED: Feed a document via the sidebar ingestion protocol first.")
    else:
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user", avatar="👤"):
            st.markdown('<div class="chat-role-label role-user">USER</div>', unsafe_allow_html=True)
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🤖"):
            st.markdown('<div class="chat-role-label role-assistant">DOCUASK AI</div>', unsafe_allow_html=True)
            with st.spinner("🤖 Neural Core Synthesizing Response..."):
                response = st.session_state.conversation_chain.invoke({
                    "question": user_input
                })
                answer = response["answer"]
                st.markdown(answer)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer
            })