# 📑 DocuAsk

**DocuAsk** is an AI-powered document Q&A application that lets you upload a document and have a conversational chat with its contents — powered by Groq's LLM and LangChain's retrieval pipeline.

---

## ✨ Features

- 📄 **Multi-format support** — Upload PDF, DOCX, DOC, TXT, and CSV files
- 🧠 **Conversational memory** — Ask follow-up questions; the app remembers your full chat history
- ⚡ **Fast inference** — Uses Groq's blazing-fast inference API
- 🔍 **Semantic search** — Documents are chunked and embedded using `sentence-transformers/all-MiniLM-L6-v2` via FAISS vector store
- 🔄 **Smart reloading** — Documents are only reprocessed when a new file is uploaded (SHA-256 hash comparison)
- 🎨 **Premium dark UI** — Custom dark theme with animated header, glassmorphism cards, and gradient accents

---

## 🖥️ Demo

> Upload a document → Ask questions → Get context-aware answers instantly.

<p align="center">
  <img src="./assets/ss1.png" alt="DocuAsk Demo" width="100%" style="border-radius: 12px; box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);" />
</p>

---

## 🛠️ Tech Stack

| Layer        | Technology                                    |
| ------------ | --------------------------------------------- |
| Frontend     | Streamlit `1.37.0`                            |
| LLM          | Groq API — `openai/gpt-oss-120b`              |
| Embeddings   | HuggingFace — `all-MiniLM-L6-v2`             |
| Vector Store | FAISS (CPU)                                   |
| Framework    | LangChain (`ConversationalRetrievalChain`)    |
| Memory       | `ConversationBufferMemory`                    |

---

## 📂 Project Structure

```
DocuAsk/
├── main.py                  # Main Streamlit application
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (not committed)
├── .streamlit/
│   └── config.toml          # Streamlit dark theme configuration
├── .devcontainer/
│   └── devcontainer.json    # GitHub Codespaces configuration
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python **3.9+**
- A valid [Groq API key](https://console.groq.com)

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
3. The Streamlit app starts automatically on port **8501** and opens in a preview pane
4. Add your `GROQ_API_KEY` to the Codespace secrets

---

## 📋 How It Works

The diagram below shows all actors and their interactions with the DocuAsk system.

```mermaid
flowchart TD
    subgraph Actors
        U(["👤 User"])
        G(["🤖 Groq LLM"])
        F(["🗄️ FAISS\nVector Store"])
    end

    subgraph DocuAsk System
        UC1["📥 Upload Document\n(PDF / DOCX / TXT / CSV)"]
        UC2["🔍 Process & Embed\nDocument"]
        UC3["💬 Ask Question"]
        UC4["🧠 Retrieve Relevant\nChunks"]
        UC5["✍️ Generate\nContext-Aware Answer"]
        UC6["🔄 Reset Session"]
        UC7["📋 View Chat History"]
    end

    %% User interactions
    U -->|"1 · Uploads file"| UC1
    U -->|"3 · Types question"| UC3
    U -->|"5 · Reads response"| UC7
    U -->|"Optionally"| UC6

    %% Internal system flow
    UC1 -->|"SHA-256 hash check\n→ only if new"| UC2
    UC2 -->|"Chunks + embeddings\nstored in"| F
    UC3 -->|"Semantic similarity\nsearch"| UC4
    UC4 <-->|"Query vectors"| F
    UC4 -->|"Top-k chunks\npassed as context"| UC5
    UC5 <-->|"LLM inference\n(ConversationalRetrievalChain)"| G
    UC5 -->|"Answer appended to"| UC7
    UC6 -->|"Clears vector store,\nchain & history"| UC2

    %% Styling
    classDef actor fill:#6C5CE7,color:#fff,stroke:#4834DF,rx:8
    classDef usecase fill:#1C1C26,color:#E2E8F0,stroke:#2D2D3E
    class U,G,F actor
    class UC1,UC2,UC3,UC4,UC5,UC6,UC7 usecase
```

### 🔑 Use Cases Explained

| # | Use Case | Actor | Description |
|---|----------|-------|-------------|
| 1 | **Upload Document** | User | Drag-and-drop or select a PDF, DOCX, TXT, or CSV file via the sidebar uploader |
| 2 | **Process & Embed Document** | System | File is chunked (1,000 chars / 200 overlap), embedded with `all-MiniLM-L6-v2`, and stored in a FAISS index. Skipped if the same file is re-uploaded (SHA-256 hash comparison) |
| 3 | **Ask Question** | User | Type a natural-language question in the chat input |
| 4 | **Retrieve Relevant Chunks** | System ↔ FAISS | Semantic similarity search returns the top-k most relevant document chunks |
| 5 | **Generate Context-Aware Answer** | System ↔ Groq LLM | `ConversationalRetrievalChain` passes retrieved chunks + full chat history to Groq's LLM and streams back the answer |
| 6 | **Reset Session** | User | Clears the FAISS index, conversation chain, and entire chat history to start fresh |
| 7 | **View Chat History** | User | All prior Q&A pairs are rendered in the chat window with full conversational memory |

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.

---

## 📜 License

This project is open source. See the repository for license details.
