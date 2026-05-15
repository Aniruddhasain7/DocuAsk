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
