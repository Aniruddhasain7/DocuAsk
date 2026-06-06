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
        model="llama-3.3-70b-versatile",
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

st.title("📑 DocuAsk")

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

uploaded_file = st.file_uploader(
    "Upload your file",
    type=["pdf", "docx", "txt", "csv"]
)

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
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                tmp.write(file_bytes)
                temp_path = tmp.name

            with st.spinner("Processing document..."):
                documents = load_document(temp_path)
                vectorstore = setup_vectorstore(documents)
                st.session_state.vectorstore = vectorstore
                st.session_state.conversation_chain = create_chain(vectorstore)
                st.session_state.chat_history = []
                st.session_state.processed_file_hash = file_hash
                st.success("Document processed successfully!")
        except Exception as e:
            st.error(f"Failed to process document: {e}")
            logging.error(f"Error loading document: {e}", exc_info=True)
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as cleanup_err:
                    logging.warning(f"Failed to clean up temporary file {temp_path}: {cleanup_err}")

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