# 📑 DocuAsk

**DocuAsk** is an AI-powered document intelligence application that lets you upload complex documents and have context-aware, multi-turn conversations with their contents — powered by Groq's high-speed inference engine, LangChain's retrieval pipeline, and ChromaDB.

---

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Open_App-00F5FF?style=for-the-badge&logo=streamlit&logoColor=black)](https://docuask.streamlit.app/)

---

## ✨ Features

- 📄 **Multi-format Document Ingestion** — Upload and analyze PDF, DOCX, DOC, TXT, and CSV files.
- 👁️ **Embedded Visual Intelligence** — Automatically detects and explains diagrams, flowcharts, UML charts, figures, and scanned pages inside PDFs and Word documents using Groq Vision (`qwen/qwen3.8-27b`).
- 🧠 **Conversational Memory** — Ask follow-up questions seamlessly with full multi-turn conversational context via `ConversationBufferMemory`.
- ⚡ **High-Speed Inference** — Sub-second response generation powered by Groq LPUs (`openai/gpt-oss-120b`).
- 🔍 **Neural Semantic Search** — High-density document chunking embedded with `sentence-transformers/all-MiniLM-L6-v2` and indexed via **ChromaDB**.
- 🔄 **Smart Session Lifecycle** — SHA-256 document hashing prevents redundant re-embedding, with one-click instant session reset.
- 🎨 **Cyberpunk Dark UI** — Custom neon HUD theme with live vector metrics, glowing telemetry badges, and glassmorphism interface cards.

---

## 🖥️ Demo

> Upload a document → Ask questions → Get context-aware answers instantly.

<p align="center">
  <img src="./assets/ss1.png" alt="DocuAsk Demo" width="100%" style="border-radius: 12px; box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);" />
</p>

---

## 🛠️ Tech Stack

| Layer            | Technology                        | Details                                              |
| :--------------- | :-------------------------------- | :--------------------------------------------------- |
| **Frontend**     | Streamlit `1.37.0`                | Custom cyberpunk dark theme & glassmorphic HUD       |
| **LLM (Chat)**   | Groq API — `openai/gpt-oss-120b`  | Ultra-fast context-grounded reasoning                |
| **Vision AI**    | Groq API — `qwen/qwen3.8-27b`     | Document diagram, chart, and visual explanation      |
| **Embeddings**   | HuggingFace — `all-MiniLM-L6-v2`  | 384-dimensional dense neural embeddings              |
| **Vector Store** | **ChromaDB** (`langchain-chroma`) | In-memory ephemeral vector database                  |
| **Framework**    | LangChain `0.2.x`                 | `ConversationalRetrievalChain`                       |
| **PDF Engine**   | `pypdf` + `pypdfium2`             | Text stream parsing & high-res visual page rendering |

---

## 📂 Project Structure

```
DocuAsk/
├── main.py                  # Main Streamlit application & RAG pipeline
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (GROQ_API_KEY)
├── .streamlit/
│   └── config.toml          # Streamlit dark theme configuration
├── .devcontainer/
│   └── devcontainer.json    # GitHub Codespaces configuration
├── assets/                  # Icons and demo preview media
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python **3.9 – 3.11**
- A free [Groq API key](https://console.groq.com)

### 1. Clone the repository

```bash
git clone https://github.com/Aniruddhasain7/DocuAsk.git
cd DocuAsk
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Run the application

```bash
streamlit run main.py
```

The app will open at **http://localhost:8501** in your browser.

---

## ☁️ Run on GitHub Codespaces

This project includes a pre-configured Dev Container for instant cloud development:

1. Click **Code → Codespaces → Create codespace on main** on GitHub
2. Wait for the container to build and dependencies to install
3. The Streamlit app starts automatically on port **8501**
4. Add your `GROQ_API_KEY` to the Codespace secrets

---

## 📋 Architecture & Data Flow

The diagram below shows all actors and their interactions with the DocuAsk system.

```mermaid
flowchart TD
    subgraph Actors
        U(["👤 User"])
        GV(["👁️ Groq Vision\n(Qwen VL)"])
        GL(["🤖 Groq LLM\n(Chat Core)"])
        C(["🗄️ ChromaDB\nVector Store"])
    end

    subgraph DocuAsk System
        UC1["📥 Upload Document\n(PDF / DOCX / TXT / CSV)"]
        UC2["🔍 Parse Text & Render\nEmbedded Visuals"]
        UC3["🧠 Generate Embeddings &\nIndex in ChromaDB"]
        UC4["💬 Ask Question"]
        UC5["⚡ Semantic Similarity\nVector Retrieval"]
        UC6["✍️ Synthesize\nContextual Answer"]
        UC7["🔄 Reset Session"]
        UC8["📋 Conversational History"]
    end

    %% Ingestion flow
    U -->|"1 · Uploads document"| UC1
    UC1 -->|"SHA-256 check (if new)"| UC2
    UC2 <-->|"Extract & explain diagrams/charts"| GV
    UC2 -->|"Text + visual explanations"| UC3
    UC3 -->|"384D vectors stored in"| C

    %% Query flow
    U -->|"2 · Types question"| UC4
    UC4 -->|"Query vector search"| UC5
    UC5 <-->|"Retrieve top-k relevant chunks"| C
    UC5 -->|"Pass chunks + chat history"| UC6
    UC6 <-->|"ConversationalRetrievalChain"| GL
    UC6 -->|"Stream response to"| UC8
    U -->|"Reads response"| UC8

    %% Reset flow
    U -.->|"Optional reset"| UC7
    UC7 -.->|"Clears Chroma collection,\nchain & history"| UC3

    %% Styling
    classDef actor fill:#6C5CE7,color:#fff,stroke:#4834DF,rx:8
    classDef usecase fill:#1C1C26,color:#E2E8F0,stroke:#2D2D3E
    class U,GV,GL,C actor
    class UC1,UC2,UC3,UC4,UC5,UC6,UC7,UC8 usecase
```

---

### 🔑 Use Cases Explained

| #   | Use Case                        | Actor                | Description                                                                                                                     |
| --- | ------------------------------- | -------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **Upload Document**             | User                 | Drag-and-drop or select a PDF, DOCX, TXT, or CSV file via the sidebar uploader.                                                 |
| 2   | **Process & Visual Extraction** | System ↔ Groq Vision | Extracts native text streams and detects embedded diagrams, charts, or scanned pages, generating thorough factual descriptions. |
| 3   | **Neural Indexing**             | System ↔ ChromaDB    | Chunks content (1,000 chars / 200 overlap), computes embeddings with `all-MiniLM-L6-v2`, and stores them in ChromaDB.           |
| 4   | **Ask Question**                | User                 | Enter natural-language queries about text, diagrams, data points, or tables.                                                    |
| 5   | **Retrieve Relevant Context**   | System ↔ ChromaDB    | Semantic similarity search returns the most relevant text and visual chunk embeddings.                                          |
| 6   | **Generate Answer**             | System ↔ Groq LLM    | `ConversationalRetrievalChain` combines retrieved context with conversation history to synthesize accurate responses.           |
| 7   | **Reset Session**               | User                 | Click `🔄 CLEAR & RESET SESSION` to purge active Chroma collections and start a new conversation.                               |
| 8   | **View Chat History**           | User                 | Prior question-and-answer pairs are rendered in chronological order with full conversational recall.                            |
