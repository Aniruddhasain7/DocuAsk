# 📑 DocuAsk

**DocuAsk** is an AI-powered document Q&A application that lets you upload a document and have a conversational chat with its contents — powered by Groq's LLaMA 3.3 70B model and LangChain.

---

## ✨ Features

- 📄 **Multi-format support** — Upload PDF, DOCX, TXT, and CSV files
- 🧠 **Conversational memory** — Ask follow-up questions; the app remembers your chat history
- ⚡ **Fast inference** — Uses Groq's blazing-fast LLaMA 3.3 70B model
- 🔍 **Semantic search** — Documents are chunked and embedded using `sentence-transformers/all-MiniLM-L6-v2` via FAISS
- 🔄 **Smart reloading** — Documents are only reprocessed when a new file is uploaded

---

## 🛠️ Tech Stack

| Layer        | Technology                       |
| ------------ | -------------------------------- |
| Frontend     | Streamlit                        |
| LLM          | Groq — `llama-3.3-70b-versatile` |
| Embeddings   | HuggingFace — `all-MiniLM-L6-v2` |
| Vector Store | FAISS                            |
| Framework    | LangChain                        |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/DocuAsk.git
cd DocuAsk
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> Get your free API key at [console.groq.com](https://console.groq.com)

### 5. Run the app

```bash
streamlit run main.py
```

The app will open at `http://localhost:8501`.

---

## 📂 Project Structure

```
DocuAsk/
├── main.py            # Main Streamlit application
├── requirements.txt   # Python dependencies
├── .env               # Environment variables (not committed)
├── .gitignore
└── README.md
```

---

## 📋 Requirements

- Python 3.9+
- A valid [Groq API key](https://console.groq.com)
