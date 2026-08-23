# Employee Handbook Assistant

A FastAPI-based RAG chatbot that answers questions from private company PDF documents using LangChain, ChromaDB, and Ollama.

## Features

- **Private data extraction** — Indexes all PDFs in the `./pdf` folder automatically on startup
- **FastAPI backend** — REST API with endpoints for chat and conversation history
- **Conversation history** — Full per-session history stored server-side
- **Agentic workflow** — LLM decides when to retrieve docs vs. calculate
- **Clean frontend** — Single-page chat UI served directly by FastAPI
- **Source attribution** — Every answer shows which document and page it came from
- **No file uploads** — PDFs are pre-loaded from the server folder

## Project Structure

```
agentic-rag/
├── main.py              # FastAPI app — routes and conversation store
├── rag_engine.py        # PDF ingestion, ChromaDB, LangChain agent
├── templates/
│   └── index.html       # Chat frontend
├── pdf/                 # Place company PDFs here
├── chroma_db/           # Auto-generated vector store (gitignored)
├── requirements.txt
└── README.md
```

## Prerequisites

1. Python 3.10+
2. [Ollama](https://ollama.com/) running locally with models pulled:

```bash
ollama pull qwen2.5:1.5b
ollama pull nomic-embed-text
```

## Installation

```bash
git clone https://github.com/varunsivasamy/agentic-rag.git
cd agentic-rag

python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

## Usage

```bash
uvicorn main:app --reload
```

Open `http://localhost:8000` in your browser.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Chat frontend |
| `POST` | `/chat` | Send a message |
| `GET` | `/history/{session_id}` | Get conversation history |
| `DELETE` | `/history/{session_id}` | Clear conversation history |
| `GET` | `/status` | Health check |

### Example `/chat` request

```json
POST /chat
{
  "session_id": "abc-123",   // omit to start a new session
  "message": "What is the leave policy?"
}
```

### Example response

```json
{
  "session_id": "abc-123",
  "reply": "Employees are entitled to 20 days of annual leave...",
  "sources": ["01_HR_Employment_Policy.pdf (p.3)"],
  "tools_used": ["document_retriever"],
  "timestamp": "2026-08-23T12:00:00"
}
```

## Adding More Documents

Drop any PDF into the `./pdf` folder and restart the server. The index rebuilds automatically when new files are detected.
